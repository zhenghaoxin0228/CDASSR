# -*- coding: utf-8 -*-
# File: train.py

import argparse
import json
import os
import random
from argparse import Namespace
from math import ceil, sqrt
from typing import Optional

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn as nn
from kornia.constants import Resample
from numpy import floating
from PIL import Image
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from loss import FFT2DLoss, MeanGradientError, MultiScaleSSIMLoss
from model import *
from test_ import test
from utils.file_ops import iter_files
from utils.num_ops import clamp
from utils.pytorch_pipeline import PyTorchPipeline


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Train CDASSR model for asymmetric arbitrary-scale SR.")
    parser.add_argument("--levels", type=str, default="BASE",
                        choices=("MOBILE", "TINY", "SMALL", "BASE", "LARGE", "XLARGE", "HUGE"))
    parser.add_argument("--block", type=str, default="ResNetBlockV2")
    parser.add_argument("--downsampler", type=str, default="maxpool2d", choices=("conv2d", "maxpool2d"))
    parser.add_argument("--upsampler", type=str, default="pixelshuffle",
                        choices=("bicubic", "bilinear", "convtranspose2d", "pixelshuffle"))
    parser.add_argument("--super_upsampler", type=str, default="scale_aware", choices=("bicubic", "scale_aware"))
    parser.add_argument("--n_recurrent", type=int, default=0)
    parser.add_argument("--channel_attention", action="store_true")
    parser.add_argument("--scale_aware_adaptation", action="store_true")
    parser.add_argument("--attention_gate", action="store_true")
    parser.add_argument("--concat_orig_interp", action="store_true")
    parser.add_argument("--layer_norm", action="store_true")
    parser.add_argument("--stochastic_depth_prob", type=float, default=0.0)
    parser.add_argument("--init_weights", action="store_true")
    parser.add_argument("--reduction", type=int, default=16)
    parser.add_argument("--n_experts", type=int, default=4)
    # Loss weights
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for L1 loss")
    parser.add_argument("--beta",  type=float, default=0.0, help="Weight for MSE loss")
    parser.add_argument("--gamma", type=float, default=0.0, help="Weight for MS-SSIM loss")
    parser.add_argument("--eta",   type=float, default=0.0, help="Weight for gradient error loss")
    parser.add_argument("--mu",    type=float, default=0.0, help="Weight for FFT loss")
    parser.add_argument("--aux_weight", type=float, default=0.01)
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # Data settings
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--max_scale", type=int, default=4)
    parser.add_argument("--asym_pct", type=float, default=0.05)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--train_dir", type=str, default="./data/train")
    parser.add_argument("--val_dir",   type=str, default="./data/valid")
    parser.add_argument("--test_dir",  type=str, default="./data/test")
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class ImageDataset(Dataset):
    """
    Loads HR images from one or more directories.
    Images are converted to float32 tensors in [0, 1].
    """
    _default_tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __init__(
        self,
        root: str | list[str] | tuple[str, ...],
        transform: Optional[v2.Transform] = None,
    ) -> None:
        super().__init__()
        roots = [root] if isinstance(root, str) else list(root)
        self.transform = transform if transform is not None else self._default_tf
        self.samples: list = []
        for r in roots:
            self.samples.extend(
                iter_files(r, exts={".png", ".jpg"}, case_insensitive=True, recursive=True)
            )

    def __getitem__(self, idx: int) -> Tensor:
        return self.transform(Image.open(self.samples[idx]))

    def __len__(self) -> int:
        return len(self.samples)


# ─────────────────────────────────────────────
# Loss Function
# ─────────────────────────────────────────────

class CDASRLoss(nn.Module):
    """
    Composite reconstruction loss with dynamically registered components.
    Only loss terms with weight > 0 are computed at runtime.
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta:  float = 0.0,
        gamma: float = 0.0,
        eta:   float = 0.0,
        mu:    float = 0.0,
    ) -> None:
        for name, val in zip(["alpha", "beta", "gamma", "eta", "mu"], [alpha, beta, gamma, eta, mu]):
            assert val >= 0.0, f"{name} must be non-negative"
        assert alpha + beta + gamma + eta + mu > 0.0, "At least one loss weight must be positive"

        super().__init__()
        _all_fns = {
            "alpha": nn.L1Loss(),
            "beta":  nn.MSELoss(),
            "gamma": MultiScaleSSIMLoss(data_range=1.0, win_size=3),
            "eta":   MeanGradientError(),
            "mu":    FFT2DLoss(),
        }
        _all_weights = {"alpha": alpha, "beta": beta, "gamma": gamma, "eta": eta, "mu": mu}

        # Register only active loss components
        self.active_fns = nn.ModuleDict(
            {k: v for k, v in _all_fns.items() if _all_weights[k] > 0.0}
        )
        self.weights = {k: w for k, w in _all_weights.items() if w > 0.0}

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return sum(self.weights[k] * fn(pred, target) for k, fn in self.active_fns.items())


# ─────────────────────────────────────────────
# Scale Sampling Utilities
# ─────────────────────────────────────────────

def build_scale_sets(max_scale: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build symmetric, asymmetric, and validation scale grids.

    Returns:
        sym_scales:   (N, 2) array of symmetric scale pairs
        asym_scales:  (M, 2) array of asymmetric scale pairs
        val_scales:   (K, 2) array of validation scale pairs (coarser grid)
    """
    base = np.arange(1.0, max_scale + 0.1, 0.1)
    sym  = np.column_stack((base, base))
    full_grid = np.array(np.meshgrid(base, base)).T.reshape(-1, 2)
    asym = np.array(list(set(map(tuple, full_grid)) - set(map(tuple, sym))))

    val_base = np.arange(1.0, max_scale + 0.3, 0.3)
    val = np.array(np.meshgrid(val_base, val_base)).T.reshape(-1, 2)
    return sym, asym, val


# ─────────────────────────────────────────────
# Data Augmentation Pipelines
# ─────────────────────────────────────────────

def build_hr_transform(img_size: int, max_scale: int) -> v2.Transform:
    """HR image augmentation applied before scale cropping."""
    return v2.Compose([
        v2.RandomCrop(img_size * max_scale, pad_if_needed=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply(nn.ModuleList([
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        ]), p=0.2),
        v2.RandomAutocontrast(p=0.2),
        v2.RandomAdjustSharpness(2, p=0.2),
        v2.RandomEqualize(p=0.2),
        v2.RandomInvert(p=0.1),
        v2.RandomGrayscale(p=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


def build_val_transform(img_size: int, max_scale: int) -> v2.Transform:
    """Deterministic center-crop transform for validation."""
    return v2.Compose([
        v2.CenterCrop(img_size * max_scale),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


def build_lr_degradation() -> K.AugmentationSequential:
    """LR degradation pipeline simulating real-world noise and blur."""
    return K.AugmentationSequential(
        K.RandomBoxBlur(kernel_size=(3, 3), p=0.125),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2), p=0.125),
        K.RandomMedianBlur(kernel_size=(3, 3), p=0.125),
        K.RandomMotionBlur(kernel_size=3, angle=180, direction=1, p=0.125),
        K.RandomGaussianNoise(p=0.25),
        K.RandomSaltAndPepperNoise(p=0.25),
        same_on_batch=False,
    )


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class CDASRTrainer(PyTorchPipeline):
    """
    Training pipeline for CDASSR.
    Handles multi-scale forward passes over both symmetric and asymmetric scale pairs.
    """

    def __init__(self, sym_scales, asym_scales, val_scales, cfg, **kwargs):
        super().__init__(**kwargs)
        self.sym_scales  = sym_scales
        self.asym_scales = asym_scales
        self.val_scales  = val_scales
        self.cfg = cfg
        self._rng = np.random.default_rng()
        self._lr_degradation = build_lr_degradation()

    def _sample_scales(self) -> np.ndarray:
        """Sample a mix of symmetric and asymmetric scales for one batch."""
        n = len(self.asym_scales)
        n_asym = clamp(int(n * self.cfg["asym_pct"]), 0, n)
        chosen_asym = self.asym_scales[
            self._rng.choice(n, n_asym, replace=False)
        ]
        return np.concatenate([self.sym_scales, chosen_asym], axis=0)

    def _forward_scale(
        self,
        hr_patch: Tensor,
        lr_input: Tensor,
        target_size: tuple[int, int],
    ) -> Tensor:
        """Single scale forward + loss computation."""
        sr_output, aux_output = self._model(lr_input, size=target_size)
        loss = self._criterion(sr_output, hr_patch)
        if self.cfg["aux_weight"] > 0:
            bicubic_ref = K.Resize(
                [self.cfg["img_size"]] * 2,
                resample=Resample.BICUBIC.name,
                antialias=True,
            )(hr_patch)
            loss = loss + self.cfg["aux_weight"] * self._criterion(aux_output, bicubic_ref)
        return loss

    # Override
    def train(self, dataloader: DataLoader, epoch: int, epochs: int) -> floating:
        self._model.train()
        accumulated_loss = []

        for hr_batch in tqdm(dataloader, desc=f"{self.get_epoch_str(epoch, epochs)} Training", leave=False):
            hr_batch = hr_batch.to(self._device)
            scale_set = self._sample_scales()

            for scale in scale_set:
                h = round(self.cfg["img_size"] * scale[0])
                w = round(self.cfg["img_size"] * scale[1])
                hr_patch = K.RandomCrop((h, w), same_on_batch=False)(hr_batch)
                bicubic_ref = K.Resize(
                    [self.cfg["img_size"]] * 2,
                    resample=random.choice((Resample.BILINEAR.name, Resample.BICUBIC.name)),
                    antialias=random.choice((True, False)),
                )(hr_patch)
                lr_input = self._lr_degradation(bicubic_ref)

                sr_output, aux_output = self._model(lr_input, size=(h, w))
                loss = self._criterion(sr_output, hr_patch)
                if self.cfg["aux_weight"] > 0:
                    loss = loss + self.cfg["aux_weight"] * self._criterion(aux_output, bicubic_ref)

                accumulated_loss.append(loss.item())
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

        return np.mean(accumulated_loss)

    # Override
    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch: int, epochs: int) -> floating:
        self._model.eval()
        accumulated_loss = []

        for hr_batch in tqdm(dataloader, desc=f"{self.get_epoch_str(epoch, epochs)} Validating", leave=False):
            hr_batch = hr_batch.to(self._device)

            for scale in self.val_scales:
                h = round(self.cfg["img_size"] * scale[0])
                w = round(self.cfg["img_size"] * scale[1])
                hr_patch   = K.CenterCrop((h, w))(hr_batch)
                bicubic_ref = K.Resize(
                    [self.cfg["img_size"]] * 2,
                    resample=Resample.BICUBIC.name,
                    antialias=True,
                )(hr_patch)

                sr_output, aux_output = self._model(bicubic_ref, size=(h, w))
                loss = self._criterion(sr_output, hr_patch)
                if self.cfg["aux_weight"] > 0:
                    loss = loss + self.cfg["aux_weight"] * self._criterion(aux_output, bicubic_ref)

                accumulated_loss.append(loss.item())

        return np.mean(accumulated_loss)


# ─────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────

def build_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    hr_tf  = build_hr_transform(args.img_size, args.max_scale)
    val_tf = build_val_transform(args.img_size, args.max_scale)

    train_loader = DataLoader(
        ImageDataset(args.train_dir, transform=hr_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        ImageDataset(args.val_dir, transform=val_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        persistent_workers=True,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print(args)

    # Model config
    model_cfg = {
        "levels":                getattr(cdassr, args.levels.upper()),
        "block":                 args.block,
        "downsampler":           args.downsampler,
        "upsampler":             args.upsampler,
        "super_upsampler":       args.super_upsampler,
        "n_recurrent":           args.n_recurrent,
        "channel_attention":     args.channel_attention,
        "scale_aware_adaptation":args.scale_aware_adaptation,
        "attention_gate":        args.attention_gate,
        "concat_orig_interp":    args.concat_orig_interp,
        "layer_norm":            args.layer_norm,
        "stochastic_depth_prob": args.stochastic_depth_prob,
        "init_weights":          args.init_weights,
        "reduction":             args.reduction,
        "n_experts":             args.n_experts,
    }

    # Trainer config (runtime hyperparams, not saved in checkpoint)
    trainer_cfg = {
        "img_size":   args.img_size,
        "asym_pct":   args.asym_pct,
        "aux_weight": args.aux_weight,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build scales
    sym_scales, asym_scales, val_scales = build_scale_sets(args.max_scale)

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(args)

    # Build model, loss, optimizer, scheduler
    model     = CDASSR(**model_cfg).to(device)
    criterion = CDASRLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, eta=args.eta, mu=args.mu)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=ceil(args.epochs / 3), gamma=sqrt(args.min_lr / args.max_lr))

    # Initialize trainer
    trainer = CDASRTrainer(
        sym_scales=sym_scales,
        asym_scales=asym_scales,
        val_scales=val_scales,
        cfg=trainer_cfg,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        name=args.name,
        configs=model_cfg,
    )

    print(f"Trainable parameters: {trainer.n_trainable_params:,}")
    output_dir = trainer.start(args.epochs, train_loader, val_dataloader=val_loader)

    # Save args
    with open(os.path.join(output_dir, "args.json"), "w") as fp:
        json.dump(vars(args), fp, indent=2)

    # Evaluate best and last checkpoints
    for tag in ("best", "last"):
        ckpt_path   = os.path.join(output_dir, f"checkpoint_{tag}.pt")
        result_file = f"test-result_{tag}.json"
        test(ckpt_path, output_dir, result_file, test_dir=args.test_dir)


if __name__ == "__main__":
    main()