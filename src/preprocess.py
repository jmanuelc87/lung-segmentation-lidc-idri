import os
import sys
import glob
import warnings

warnings.filterwarnings("ignore", module="pylidc", category=UserWarning)


import numpy as np
import pandas as pd
import pylidc as pl
import SimpleITK as sitk
import concurrent.futures as c
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from pylidc.utils import consensus
from datasets import Dataset, Image as HFImage
from collections import deque
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv
from statistics import median_high


load_dotenv(find_dotenv())


IMG_MASK = "./data/image_masks"
PATCH_MASK = "./data/patches_mask"
IMG = "./data/images"
PATCH = "./data/patches"


class NoP:

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def save_image(arr: np.ndarray, file_path: str, interpolator=sitk.sitkLinear):
    image = sitk.GetImageFromArray(arr)

    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    new_spacing = [(original_size[0] - 1) * original_spacing[0] / (512 - 1)] * 3
    new_size = [
        512,
        int((original_size[1] - 1) * original_spacing[1] / new_spacing[1]),
        1,
    ]

    image = sitk.Resample(
        image1=image,
        size=new_size,  # type: ignore
        transform=sitk.Transform(),
        interpolator=interpolator,
        outputOrigin=image.GetOrigin(),
        outputSpacing=new_spacing,
        outputDirection=image.GetDirection(),
        defaultPixelValue=0,
        outputPixelType=image.GetPixelID(),
    )

    if image.GetNumberOfComponentsPerPixel() == 1:
        image = sitk.RescaleIntensity(image, 0, 255)
        image = sitk.Cast(image, sitk.sitkUInt8)
        sitk.WriteImage(image, file_path)


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_image_patch(id, i, vol, anns, pad, mask, image):
    cmask, cbbox = consensus(  # type: ignore
        anns, clevel=0.5, pad=[(pad, pad), (pad, pad), (0, 0)], ret_masks=False
    )
    k = int(0.5 * (cbbox[2].stop - cbbox[2].start))
    arr = vol[cbbox][:, :, k].astype(np.float32)

    save_image(
        cmask[:, :, k].astype(np.float32),
        os.path.join(mask, f"{id:04d}-{i:02d}.png"),
        interpolator=sitk.sitkNearestNeighbor,
    )
    save_image(arr, os.path.join(image, f"{id:04d}-{i:02d}.png"))


def preprocess(scan, coll: deque, progress):
    with NoP():
        vol = scan.to_volume()
        nods = scan.cluster_annotations()

    check_dir(IMG_MASK)
    check_dir(IMG)
    check_dir(PATCH_MASK)
    check_dir(PATCH)

    for i, anns in enumerate(nods):
        # IMAGES
        create_image_patch(scan.id, i, vol, anns, pad=512, mask=IMG_MASK, image=IMG)

        # PATCHES
        create_image_patch(scan.id, i, vol, anns, pad=48, mask=PATCH_MASK, image=PATCH)

        # LABELS
        list = []
        for ann in anns:
            list.append(ann.malignancy)

        labels = {}
        malignancy = median_high(list)
        labels["malignancy"] = malignancy
        if malignancy > 3:
            labels["cancer"] = "Yes"
        elif malignancy < 3:
            labels["cancer"] = "No"
        else:
            labels["cancer"] = "Ambiguous"

        coll.append({"id": f"{scan.id:04d}-{i:02d}", "label": labels})
    progress.update()


def create_dataset():
    patch_paths = glob.glob(os.path.join(PATCH, "*.png"))
    patch_mask_paths = glob.glob(os.path.join(PATCH_MASK, "*.png"))
    image_paths = glob.glob(os.path.join(IMG, "*.png"))
    image_mask_paths = glob.glob(os.path.join(IMG_MASK, "*.png"))
    labels = pd.read_csv("./data/labels.csv")

    dataset = Dataset.from_dict(
        {
            "image": sorted(image_paths),
            "image_mask": sorted(image_mask_paths),
            "patch": sorted(patch_paths),
            "patch_mask": sorted(patch_mask_paths),
            "malignancy": [el["malignancy"] for _, el in labels.iterrows()],
            "cancer": [el["cancer"] for _, el in labels.iterrows()],
        }
    )

    dataset = dataset.cast_column("image", HFImage())
    dataset = dataset.cast_column("image_mask", HFImage())
    dataset = dataset.cast_column("patch", HFImage())
    dataset = dataset.cast_column("patch_mask", HFImage())

    login(token=os.environ.get("HF_TOKEN"))
    dataset.push_to_hub("lidc-idri-segmentation")


def main():
    scans = pl.query(pl.Scan).all()
    count = pl.query(pl.Scan).count()

    progress = tqdm(total=count)

    results = []
    coll = deque()
    with c.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for scan in scans:
            result = executor.submit(preprocess, scan, coll, progress)

        for result in results:
            result.join()

    labels = [
        {
            "id": el["id"],
            "malignancy": el["label"]["malignancy"],
            "cancer": el["label"]["cancer"],
        }
        for el in sorted(list(coll), key=lambda x: x["id"])
    ]

    df = pd.DataFrame(data=labels)
    df.to_csv("./data/labels.csv")
