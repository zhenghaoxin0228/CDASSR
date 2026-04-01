import math
from typing import Any, Optional, Type

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops.stochastic_depth import StochasticDepth


class LayerNorm2d(nn.LayerNorm):

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class ChannelModification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modification = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.modification(x)


class Concatenation(nn.Module):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, *inputs: Tensor) -> Tensor:
        return torch.cat(inputs, dim=self.dim)


class ChannelAttention(nn.Module):

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        reduced = max(channels // reduction, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.attention(x)


    class SAFA(nn.Module):
        def __init__(self, channels, kernel_size=3, n_experts=4, dropout=0.5):
            super().__init__()
            self.channels = channels
            self.kernel_size = kernel_size
            self.n_experts = n_experts

            self.routing = nn.Sequential(
                nn.Linear(2, n_experts * 8),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(n_experts * 8, n_experts),
                nn.Softmax(dim=1),
            )
            self.weight_pool = nn.Parameter(Tensor(n_experts, channels, 1, kernel_size, kernel_size))
            nn.init.trunc_normal_(self.weight_pool, std=0.02)
            self.bias_pool = nn.Parameter(Tensor(n_experts, channels))
            nn.init.constant_(self.bias_pool, 0)

            self.gate = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
                nn.Sigmoid(),
            )

        def forward(self, x, scale_h, scale_w):
            scale_h_t = torch.ones(1, 1).to(x.device) / scale_h
            scale_w_t = torch.ones(1, 1).to(x.device) / scale_w

            routing_weights = self.routing(
                torch.cat((scale_h_t, scale_w_t), 1)
            ).view(self.n_experts, 1, 1)
            fused_weight = (self.weight_pool.view(self.n_experts, -1, 1) * routing_weights).sum(0)
            fused_weight = fused_weight.view(-1, 1, self.kernel_size, self.kernel_size)
            fused_bias = (self.bias_pool.view(self.n_experts, -1, 1) * routing_weights).sum(0)
            fused_bias = fused_bias.view(-1)
            F_ad = F.conv2d(x, fused_weight, fused_bias, padding="same", groups=self.channels)

            G = self.gate(x)

            return F_ad * G


    def forward(self, x, scale_h, scale_w):
        scale_h_t = torch.ones(1, 1).to(x.device) / scale_h
        scale_w_t = torch.ones(1, 1).to(x.device) / scale_w

        # 生成 scale-aware 动态卷积核并应用 → F_ad
        routing_weights = self.routing(
            torch.cat((scale_h_t, scale_w_t), 1)
        ).view(self.n_experts, 1, 1)
        fused_weight = (self.weight_pool.view(self.n_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, 1, self.kernel_size, self.kernel_size)
        fused_bias = (self.bias_pool.view(self.n_experts, -1, 1) * routing_weights).sum(0)
        fused_bias = fused_bias.view(-1)
        F_ad = F.conv2d(x, fused_weight, fused_bias, padding="same", groups=self.channels)

        # 生成局部结构门控图 G
        G = self.gate(x)

        # SAFA 融合：F_i' = F_ad × G（替换原来的 x + adapted * mask）
        return F_ad * G

class RecurrentAttentionBlock(nn.Module):

    def __init__(
        self,
        block: Type[nn.Module],
        channels: int,
        n_recurrent: int = 0,
        attention: bool = False,
        scale_aware: bool = False,
        layer_norm: bool = False,
        stochastic_depth_prob: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.block = block
        self.channels = channels
        self.n_recurrent = n_recurrent
        self.attention = attention
        self.scale_aware = scale_aware
        self.layer_norm = layer_norm
        self.stochastic_depth_prob = stochastic_depth_prob
        self.reduction: int = kwargs.pop("reduction", 16)
        self.n_experts: int = kwargs.pop("n_experts", 4)
        self.eps: float = kwargs.pop("eps", 1e-6)

        self.bottleneck = block(channels, layer_norm=layer_norm, eps=self.eps, **kwargs)
        if attention:
            self.channel_attention = ChannelAttention(channels, reduction=self.reduction)
        if scale_aware:
            self.scale_aware_adaptation = SAFA(channels, n_experts=self.n_experts)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(
        self,
        x: Tensor,
        scale_h: Optional[float] = None,
        scale_w: Optional[float] = None,
    ) -> Tensor:
        if self.scale_aware:
            assert scale_h is not None and scale_w is not None

        out = self.bottleneck(x)

        for _ in range(self.n_recurrent):
            out = self.bottleneck(x + out)

        if self.attention:
            out = self.channel_attention(out)

        out = x + self.stochastic_depth(out)

        if self.scale_aware:
            out = self.scale_aware_adaptation(out, scale_h, scale_w)

        return out


class GRN(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        Gx = x.norm(p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class AttentionGate(nn.Module):

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.attention = nn.Sequential(
            nn.GELU(),
            ChannelModification(channels, 1),
            LayerNorm2d(1, eps=eps),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        assert x.shape[1] == self.channels
        assert g.shape[1] == self.channels

        return x * self.attention(x + g)


def grid_sample(
    x: Tensor,
    offset: Tensor,
    out_h: int,
    out_w: int,
) -> Tensor:

    b, _, h, w = x.size()
    scale_h, scale_w = out_h / h, out_w / w

    # generate grids
    grid = np.meshgrid(range(out_w), range(out_h))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale_w - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale_h - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) - 1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) - 1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    grid = grid + torch.cat((offset_0, offset_1), 1)
    grid = grid.permute(0, 2, 3, 1)

    # sampling
    output = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

    return output


class LASU(nn.Module):
    def __init__(self, channels, n_experts=4, reduction=16, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.n_experts = n_experts
        self.reduction = reduction
        self.eps = eps
        self.reduced = max(channels // reduction, 1)

        self.weight_compress = nn.Parameter(...)
        self.weight_expand   = nn.Parameter(...)
        self.body     = nn.Sequential(...)
        self.routing  = nn.Sequential(...)
        self.offset   = nn.Conv2d(64, 2, 1)

        self.se_block = ChannelAttention(channels, reduction=reduction)
        self.branch1_conv = nn.Conv2d(channels, channels, kernel_size=3, padding="same")

        self.fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x, out_h, out_w):
        b, _, h, w = x.shape
        F_da = self.branch1_conv(self.se_block(x))  # (B, C, H, W)
        F_da = F.interpolate(F_da, size=(out_h, out_w), mode='bilinear', align_corners=False)

        scale_h, scale_w = out_h / h, out_w / w
        fea0 = grid_sample(x, offset, out_h, out_w)
        fea  = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)
        out  = torch.matmul(weight_compress.expand([b,-1,-1,-1,-1]), fea)
        F_ss = torch.matmul(weight_expand.expand([b,-1,-1,-1,-1]), out).squeeze(-1)
        F_ss = F_ss.permute(0, 3, 1, 2) + fea0  # (B, C, out_h, out_w)

        F_hybrid = torch.cat([F_da, F_ss], dim=1)  # (B, 2C, out_h, out_w)
        return self.fusion_conv(F_hybrid)           # (B, C, out_h, out_w)