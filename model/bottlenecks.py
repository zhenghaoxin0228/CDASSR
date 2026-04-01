from typing import Any

import torch
from torch import Tensor, nn

from .utils import GRN, LayerNorm2d

class ResNetBlockV2(nn.Module):


    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        layer_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.expansion = expansion
        self.layer_norm = layer_norm
        self.eps: float = kwargs.pop("eps", 1e-6)

        expanded = channels * expansion
        self.block = nn.Sequential(
            LayerNorm2d(channels, eps=self.eps) if layer_norm else nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding="same", bias=layer_norm),
            LayerNorm2d(channels, eps=self.eps) if layer_norm else nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, expanded, kernel_size=1, bias=layer_norm),
            LayerNorm2d(expanded, eps=self.eps) if layer_norm else nn.BatchNorm2d(expanded),
            nn.GELU(),
            nn.Conv2d(expanded, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        layer_scale_init_value: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.expansion = expansion
        self.layer_scale_init_value = layer_scale_init_value
        self.eps: float = kwargs.pop("eps", 1e-6)

        expanded = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding="same", groups=channels),
            LayerNorm2d(channels, eps=self.eps),
            nn.Conv2d(channels, expanded, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(expanded, channels, kernel_size=1),
        )
        self.layer_scale = nn.Parameter(torch.ones(channels, 1, 1) * layer_scale_init_value)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x) * self.layer_scale


class ConvNeXtBlockV2(nn.Module):

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.expansion = expansion
        self.eps: float = kwargs.pop("eps", 1e-6)

        expanded = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding="same", groups=channels),
            LayerNorm2d(channels, eps=self.eps),
            nn.Conv2d(channels, expanded, kernel_size=1),
            nn.GELU(),
            GRN(expanded),
            nn.Conv2d(expanded, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class EDSRBlock(nn.Module):

    def __init__(self, channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.channels = channels

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class TwiResBlock(nn.Module):

    def __init__(self, channels: int, layer_norm: bool = False, **kwargs):
        super().__init__()
        self.res1 = ResNetBlockV2(channels, layer_norm=layer_norm, **kwargs)
        self.res2 = ResNetBlockV2(channels, layer_norm=layer_norm, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.res2(self.res1(x))