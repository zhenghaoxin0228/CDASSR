# -*- coding: utf-8 -*-
# File: loss/fft.py

import torch
from torch import Tensor, nn


class FFT2DLoss(nn.Module):
    """
    References: [Focal Frequency Loss for Image Reconstruction and Synthesis](https://arxiv.org/abs/2012.12821),
    [Fourier Space Losses for Efficient Perceptual Image Super-Resolution](https://arxiv.org/abs/2106.00783)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        f_pred = torch.fft.fft2(pred, norm="ortho")
        f_target = torch.fft.fft2(target, norm="ortho")

        return torch.mean(torch.abs(f_pred - f_target))

    @staticmethod
    def visualize(img: Tensor, eps: float = 1e-6) -> None:
        import matplotlib.pyplot as plt
        from torchvision.transforms.v2 import Grayscale

        assert 2 <= len(img.shape) <= 3
        if len(img.shape) == 3:
            if img.shape[0] == 3:
                img = Grayscale()(img)
            img = img[0]

        f = torch.fft.fftshift(torch.fft.fft2(img, norm="ortho"))
        f = 20 * torch.log10(torch.abs(f) + eps)
        f = (f - torch.min(f)) / (torch.max(f) - torch.min(f))

        plt.imshow(f, cmap="gray")
        plt.show()
        plt.close()
