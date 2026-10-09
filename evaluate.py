#!/usr/bin/env python

from email.mime import image
import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.linalg import sqrtm

from skimage.metrics import structural_similarity
from skimage.feature import graycomatrix, graycoprops
from skimage import filters
from sklearn.metrics import mutual_info_score

from tifffile import imread
from PIL import Image

from tqdm.auto import tqdm

import argparse

PRED_SUFFIX = ".tiff"
TRUTH_SUFFIX = "_protein.png"


def image_loader(path):
    if path.endswith(".tiff"):
        img = imread(path).astype(np.float32)
    else:
        with Image.open(path) as source:
            if source.format != "PNG" or source.mode != "L":
                raise ValueError(f"PNG must be 8-bit grayscale: {path.name}")
            img = np.asarray(source, dtype=np.float32)
    return img

def get_image_pairs(truth_dir, predictions_dir):
    truth_files = sorted(f for f in os.listdir(truth_dir) if f.endswith(TRUTH_SUFFIX))
    pairs = []
    for fname in truth_files:
        pair_name = fname.replace(TRUTH_SUFFIX, "")

        pred_fname = fname.replace(TRUTH_SUFFIX, PRED_SUFFIX)
        pred_path  = os.path.join(predictions_dir, pred_fname)
        if not os.path.exists(pred_path):
            print(f"  Missing prediction for {fname}, skipping")
            continue
        truth_path = os.path.join(truth_dir, fname)

        img_real = image_loader(truth_path)
        img_pred = image_loader(pred_path)

        # Skip constant images (undefined correlation / texture)
        if np.std(img_real) == 0 or np.std(img_pred) == 0:
            print(f"  Skipping constant image: {fname}")
            continue
        pairs.append({"name": pair_name, "real": img_real, "pred": img_pred})
    return pairs

def pixel_correlations(img_real, img_pred):
    flat_r = img_real.flatten()
    flat_p = img_pred.flatten()
    pearson  = stats.pearsonr(flat_r, flat_p)[0]
    spearman = stats.spearmanr(flat_r, flat_p).statistic
    return pearson, spearman


def compute_psnr(img_real, img_pred, data_range=1.0):
    mse = np.mean((img_real - img_pred) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(data_range) - 10 * np.log10(mse)


def compute_ms_ssim(img_real, img_pred, win_size=11, data_range=1.0):
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    img1, img2 = img_real.astype(np.float64), img_pred.astype(np.float64)
    mcs, mssim = [], []
    for scale in range(len(weights)):
        sim, cs_map = structural_similarity(img1, img2, win_size=win_size,
                                            data_range=data_range, full=True)
        mssim.append(sim); mcs.append(np.mean(cs_map))
        if scale < len(weights) - 1:
            if min(img1.shape) >= win_size * 2:
                h, w = img1.shape
                img1 = img1[:h//2*2, :w//2*2]; img2 = img2[:h//2*2, :w//2*2]
                img1 = (img1[::2,::2]+img1[1::2,::2]+img1[::2,1::2]+img1[1::2,1::2])/4
                img2 = (img2[::2,::2]+img2[1::2,::2]+img2[::2,1::2]+img2[1::2,1::2])/4
            else:
                weights = weights[:scale+1] / weights[:scale+1].sum(); break
    eps = 1e-8
    mcs = np.clip(mcs, eps, 1.0); mssim = np.clip(mssim, eps, 1.0)
    val = np.prod(mcs[:-1]**weights[:-1]) * mssim[-1]**weights[-1] if len(mcs)>1 else mssim[0]
    return float(np.clip(val, 0.0, 1.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image pairs")
    parser.add_argument("predictions_dir", type=str, help="Folder containing *_protein.png")
    parser.add_argument("truth_dir", type=str, help="Folder containing ground *_protein.png")
    parser.add_argument("-o", "--output", type=str, default="stats.json", help="Output JSON file to store evaluation results")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")


    pairs = get_image_pairs(args.truth_dir, args.predictions_dir)
    print(f"Loaded {len(pairs)} image pairs")
    with open(args.output, "w") as f:
        for pair in tqdm(pairs):
            fname = pair["name"]
            img_real = pair["real"]
            img_pred = pair["pred"]
            pearson, spearman = pixel_correlations(img_real, img_pred)
            psnr = compute_psnr(img_real, img_pred)
            ms_ssim = compute_ms_ssim(img_real, img_pred)
            result = {
                "name": fname,
                "pearson": float(pearson),
                "spearman": float(spearman),
                "psnr": float(psnr),
                "ms_ssim": float(ms_ssim)
            }
            json.dump(result, f)
            f.write("\n")
 