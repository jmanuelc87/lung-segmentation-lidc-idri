import os
import glob
import queue
import numpy as np
import pylidc as pl
import threading as t
import concurrent.futures as c
import matplotlib.pyplot as plt

from PIL import Image
from pylidc.utils import consensus
from datasets import Dataset, Image as HFImage
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


def process(scan):
    vol = scan.to_volume()
    nods = scan.cluster_annotations()

    for i, anns in enumerate(nods):
        cmask, cbbox = consensus(  # type: ignore
            anns, clevel=0.5, pad=[(512, 512), (512, 512), (0, 0)], ret_masks=False
        )
        k = int(0.5 * (cbbox[2].stop - cbbox[2].start))

        image = Image.fromarray(cmask[:, :, k])
        image.save(f"./data/masks/{scan.id:04d}-{i:02d}.png")

        arr = vol[cbbox][:, :, k].astype(np.float32)

        hu_min = -1000  # lower bound (air)
        hu_max = 400  # upper bound (soft tissue)

        arr[arr <= -2000] = hu_min  # replace padding with air

        arr = np.clip(arr, hu_min, hu_max)  # clamp
        arr = (arr - hu_min) / (hu_max - hu_min)  # [0,1]
        arr = (arr * 255).astype(np.uint8)  # [0,255] uint8 for image

        image = Image.fromarray(arr)
        image.save(f"./data/images/{scan.id:04d}-{i:02d}.png")


def create_dataset():
    image_paths = glob.glob("data/images/*.png")
    mask_paths = glob.glob("data/masks/*.png")

    dataset = Dataset.from_dict(
        {
            "image": sorted(image_paths),
            "mask": sorted(mask_paths),
        }
    )

    dataset = dataset.cast_column("image", HFImage())
    dataset = dataset.cast_column("mask", HFImage())

    login(token=os.environ.get("HF_TOKEN"))

    dataset.push_to_hub("lidc-idri-segmentation")


def main():
    scans = pl.query(pl.Scan).all()

    results = []
    with c.ThreadPoolExecutor(max_workers=12) as executor:
        for scan in scans:
            result = executor.submit(process, scan)

        for result in results:
            result.join()

    print("Done")
