# -*- coding: utf-8 -*-
# File: test_.py

import argparse
import json
import os
from argparse import Namespace
from statistics import mean
from typing import Optional

import torch
from kornia.color import rgb_to_ycbcr
from PIL import Image
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode
from tqdm import tqdm

from infer import CDASRInferencer
from utils.file_ops import iter_files
from utils.pytorch_pipeline import PyTorchPipeline


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CDASSR model on test datasets.")
    parser.add_argument("model_path",  type=str, help="Path to CDASSR checkpoint (.pt).")
    parser.add_argument("output_dir",  type=str, help="Directory to save evaluation results.")
    parser.add_argument("result_json", type=str, help="Filename for the output JSON report.")
    parser.add_argument("--test_dir",  type=str, default="./data/test")
    parser.add_argument("--scales", nargs="+", type=float, default=(2, 3, 4),
                        help="Upscaling factors to evaluate.")
    parser.add_argument("--border", type=int, default=None,
                        help="Number of border pixels to crop before computing metrics.")
    parser.add_argument("--save", action="store_true",
                        help="Whether to save upscaled output images.")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Metric Helpers
# ─────────────────────────────────────────────

def _crop_border(t: Tensor, border: Optional[int]) -> Tensor:
    """Crop a fixed border from a 4-D tensor (B, C, H, W)."""
    if border is None:
        return t
    return t[:, :, border:-border, border:-border]


def _compute_metrics(
    pred: Tensor,
    target: Tensor,
    psnr_fn: PeakSignalNoiseRatio,
    ssim_fn: StructuralSimilarityIndexMeasure,
    border: Optional[int],
) -> dict[str, float]:
    """
    Compute PSNR and SSIM on both RGB and Y channels.
    Returns a dict with keys: PSNR_RGB, PSNR_Y, SSIM_RGB, SSIM_Y.
    """
    pred_rgb   = _crop_border(pred,   border)
    target_rgb = _crop_border(target, border)

    pred_y   = rgb_to_ycbcr(pred)[:, :1]
    target_y = rgb_to_ycbcr(target)[:, :1]

    return {
        "PSNR_RGB": psnr_fn(pred_rgb,   target_rgb).item(),
        "PSNR_Y":   psnr_fn(pred_y,     target_y).item(),
        "SSIM_RGB": ssim_fn(pred_rgb,   target_rgb).item(),
        "SSIM_Y":   ssim_fn(pred_y,     target_y).item(),
    }


# ─────────────────────────────────────────────
# Per-dataset Evaluation
# ─────────────────────────────────────────────

def _evaluate_dataset(
    dataset_path: str,
    dataset_name: str,
    scales: list[float],
    inferencer: CDASRInferencer,
    psnr_fn: PeakSignalNoiseRatio,
    ssim_fn: StructuralSimilarityIndexMeasure,
    to_tensor: v2.Transform,
    border: Optional[int],
    save: bool,
    save_dir: Optional[str],
) -> dict:
    """
    Evaluate all images in a single dataset folder across all requested scales.
    Returns per-scale averaged metrics.
    """
    metric_keys = ["PSNR_RGB", "PSNR_Y", "SSIM_RGB", "SSIM_Y"]
    # Accumulate per-image scores
    per_scale: dict[float, dict[str, list]] = {
        s: {k: [] for k in metric_keys} for s in scales
    }

    image_paths = list(
        iter_files(dataset_path, exts={".jpg", ".png"}, case_insensitive=True, recursive=True)
    )

    for img_path in tqdm(image_paths, desc=dataset_name, leave=False):
        hr: Tensor = to_tensor(Image.open(img_path).convert("RGB"))
        hr = hr.unsqueeze(0).to(psnr_fn.device)
        hr_h, hr_w = hr.shape[-2:]

        for scale in scales:
            # Downsample HR to LR
            lr_h, lr_w = hr_h // scale, hr_w // scale
            lr = v2.Resize(
                (lr_h, lr_w),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )(hr)

            # Upscale back to HR resolution
            sr = inferencer(img_tensor=lr, size=(hr_h, hr_w))

            # Optionally save output
            if save and save_dir is not None:
                out_folder = os.path.join(save_dir, dataset_name, f"X{scale}")
                os.makedirs(out_folder, exist_ok=True)
                v2.ToPILImage()(sr.squeeze()).save(
                    os.path.join(out_folder, img_path.name)
                )

            # Accumulate metrics
            scores = _compute_metrics(sr, hr, psnr_fn, ssim_fn, border)
            for k, v in scores.items():
                per_scale[scale][k].append(v)

    # Average across all images
    averaged: dict[float, dict[str, float]] = {}
    for scale in scales:
        averaged[scale] = {k: mean(per_scale[scale][k]) for k in metric_keys}

    return averaged


# ─────────────────────────────────────────────
# Main Evaluation Function
# ─────────────────────────────────────────────

def test(
    model_path: str,
    output_dir: str,
    result_json: str,
    test_dir: str = "./data/test",
    scales: list[float] = (2, 3, 4),
    border: Optional[int] = 4,
    save: bool = False,
) -> None:
    """
    Run full evaluation over all sub-datasets in test_dir.
    Saves a JSON report with per-dataset and overall mean metrics.
    """
    if border is not None and border <= 0:
        raise ValueError("`border` must be None or a positive integer.")

    os.makedirs(output_dir, exist_ok=True)
    save_dir = None
    if save:
        save_dir = os.path.join(output_dir, "sr_outputs")
        os.makedirs(save_dir, exist_ok=True)

    device     = PyTorchPipeline.get_device()
    inferencer = CDASRInferencer(model_path, device=device)
    to_tensor  = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    psnr_fn    = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn    = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    metric_keys = ["PSNR_RGB", "PSNR_Y", "SSIM_RGB", "SSIM_Y"]
    global_scores: dict[str, list] = {k: [] for k in metric_keys}
    report: dict = {"mean": {k: 0.0 for k in metric_keys}}

    for entry in tqdm(os.listdir(test_dir), desc="Evaluating datasets"):
        dataset_path = os.path.join(test_dir, entry)
        if not os.path.isdir(dataset_path):
            continue

        dataset_metrics = _evaluate_dataset(
            dataset_path=dataset_path,
            dataset_name=entry,
            scales=list(scales),
            inferencer=inferencer,
            psnr_fn=psnr_fn,
            ssim_fn=ssim_fn,
            to_tensor=to_tensor,
            border=border,
            save=save,
            save_dir=save_dir,
        )
        report[entry] = dataset_metrics

        # Accumulate for global mean
        for scale in scales:
            for k in metric_keys:
                global_scores[k].append(dataset_metrics[scale][k])

    # Compute global mean across all datasets and scales
    report["mean"] = {k: mean(global_scores[k]) for k in metric_keys}

    # Persist report
    report_path = os.path.join(output_dir, result_json)
    with open(report_path, "w") as fp:
        json.dump(report, fp, indent=2)

    print(f"Evaluation complete. Results saved to: {report_path}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    test(
        model_path=args.model_path,
        output_dir=args.output_dir,
        result_json=args.result_json,
        test_dir=args.test_dir,
        scales=args.scales,
        border=args.border,
        save=args.save,
    )