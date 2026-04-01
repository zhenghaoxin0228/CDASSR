# -*- coding: utf-8 -*-
# File: loss/mge.py

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MeanGradientError(nn.Module):
    """
    Reference: [Single Image Super Resolution based on a Modified U-net with Mixed Gradient Loss](https://arxiv.org/abs/1911.09428)
    """

    SOBEL_X = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    SOBEL_Y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        filter_x = torch.tensor(self.SOBEL_X, dtype=pred.dtype, device=pred.device).repeat(pred.shape[1], 1, 1, 1)
        filter_y = torch.tensor(self.SOBEL_Y, dtype=pred.dtype, device=pred.device).repeat(pred.shape[1], 1, 1, 1)

        pred_gx = F.conv2d(pred, filter_x, padding="same", groups=pred.shape[1])
        pred_gy = F.conv2d(pred, filter_y, padding="same", groups=pred.shape[1])
        target_gx = F.conv2d(target, filter_x, padding="same", groups=pred.shape[1])
        target_gy = F.conv2d(target, filter_y, padding="same", groups=pred.shape[1])
        pred_gmag = torch.sqrt(pred_gx**2 + pred_gy**2 + self.eps)
        target_gmag = torch.sqrt(target_gx**2 + target_gy**2 + self.eps)

        return F.mse_loss(pred_gmag, target_gmag)

    @staticmethod
    def visualize(img: Tensor) -> None:
        import matplotlib.pyplot as plt
        from torchvision.transforms.v2 import Grayscale

        assert 2 <= len(img.shape) <= 3
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        elif img.shape[0] == 3:
            img = Grayscale()(img)

        filter_x = torch.tensor(MeanGradientError.SOBEL_X, dtype=img.dtype, device=img.device).repeat(1, 1, 1, 1)
        filter_y = torch.tensor(MeanGradientError.SOBEL_Y, dtype=img.dtype, device=img.device).repeat(1, 1, 1, 1)

        gx = F.conv2d(img, filter_x, padding="same")
        gy = F.conv2d(img, filter_y, padding="same")
        gmag = torch.sqrt(gx**2 + gy**2)
        gmag = (gmag - torch.min(gmag)) / (torch.max(gmag) - torch.min(gmag))

        plt.imshow(gmag[0], cmap="gray")
        plt.show()
        plt.close()
