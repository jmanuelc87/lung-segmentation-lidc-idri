import os
import glob
import numpy as np
import pylidc as pl
import SimpleITK as sitk
import threading as t
import concurrent.futures as c
import matplotlib.pyplot as plt

from PIL import Image
from pylidc.utils import consensus
from datasets import Dataset, Image as HFImage
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

IMG_MASK = "./data/image_masks"
PATCH_MASK = "./data/patches_mask"
IMG = "./data/images"
PATCH = "./data/patches"


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


def preprocess(scan):
    vol = scan.to_volume()
    nods = scan.cluster_annotations()

    check_dir(IMG_MASK)
    check_dir(IMG)
    check_dir(PATCH_MASK)
    check_dir(PATCH)

    for i, anns in enumerate(nods):
        # IMAGES
        cmask, cbbox = consensus(  # type: ignore
            anns, clevel=0.5, pad=[(512, 512), (512, 512), (0, 0)], ret_masks=False
        )
        k = int(0.5 * (cbbox[2].stop - cbbox[2].start))
        arr = vol[cbbox][:, :, k].astype(np.float32)

        save_image(
            cmask[:, :, k].astype(np.float32),
            os.path.join(IMG_MASK, f"{scan.id:04d}-{i:02d}.png"),
            interpolator=sitk.sitkNearestNeighbor,
        )
        save_image(arr, os.path.join(IMG, f"{scan.id:04d}-{i:02d}.png"))

        # PATCHES
        cmask, cbbox = consensus(  # type: ignore
            anns, clevel=0.5, pad=[(48, 48), (48, 48), (0, 0)], ret_masks=False
        )
        k = int(0.5 * (cbbox[2].stop - cbbox[2].start))
        arr = vol[cbbox][:, :, k].astype(np.float32)

        save_image(
            cmask[:, :, k].astype(np.float32),
            os.path.join(PATCH_MASK, f"{scan.id:04d}-{i:02d}.png"),
            interpolator=sitk.sitkNearestNeighbor,
        )
        save_image(arr, os.path.join(PATCH, f"{scan.id:04d}-{i:02d}.png"))


def create_dataset():
    patch_paths = glob.glob(os.path.join(PATCH, "*.png"))
    patch_mask_paths = glob.glob(os.path.join(PATCH_MASK, "*.png"))
    image_paths = glob.glob(os.path.join(IMG, "*.png"))
    image_mask_paths = glob.glob(os.path.join(IMG_MASK, "*.png"))

    dataset = Dataset.from_dict(
        {
            "image": sorted(image_paths),
            "image_mask": sorted(image_mask_paths),
            "patch": sorted(patch_paths),
            "patch_mask": sorted(patch_mask_paths),
        }
    )

    dataset = dataset.cast_column("image", HFImage())
    dataset = dataset.cast_column("image_mask", HFImage())
    dataset = dataset.cast_column("patch", HFImage())
    dataset = dataset.cast_column("patch_mask", HFImage())

    login(token=os.environ.get("HF_TOKEN"))
    dataset.push_to_hub("lidc-idri-segmentation")
    print("Done")


def main():
    scans = pl.query(pl.Scan).all()

    results = []
    with c.ThreadPoolExecutor(max_workers=12) as executor:
        for scan in scans:
            result = executor.submit(preprocess, scan)

        for result in results:
            result.join()

    print("Done")
