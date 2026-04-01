from collections.abc import Sequence
from typing import Any, Literal, Type

from torch import Tensor, nn

from . import bottlenecks
from .utils import (
    AttentionGate,
    ChannelModification,
    Concatenation,
    LayerNorm2d,
    RecurrentAttentionBlock,
    LASU,
)

__all__ = [
    "CDASSR",
    "MOBILE",
    "TINY",
    "SMALL",
    "BASE",
    "LARGE",
    "XLARGE",
    "HUGE",
]

MOBILE = ((16, 3), (32, 3))  # ~64K parameters
TINY = ((32, 6), (64, 12), (128, 6))  # ~1.6M parameters
SMALL = ((64, 3), (128, 3), (256, 9), (512, 3))  # ~19M parameters
BASE = ((64, 9), (128, 9), (256, 27), (512, 9))  # ~43M parameters
LARGE = ((96, 9), (192, 9), (384, 27), (768, 9))  # ~95M parameters
XLARGE = ((96, 15), (192, 15), (384, 45), (768, 15))  # ~149M parameters
HUGE = ((96, 12), (192, 12), (384, 24), (768, 36), (1536, 12))  # ~548M parameters


class Downsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int,
        mode: Literal["conv2d", "maxpool2d"] = "conv2d",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.mode = mode
        self.eps = eps

        if scale == 1:
            self.downsampler = nn.Identity()
        else:
            if mode == "conv2d":
                self.downsampler = nn.Sequential(
                    LayerNorm2d(in_channels, eps=eps),
                    nn.Conv2d(in_channels, out_channels, kernel_size=scale, stride=scale),
                )
            elif mode == "maxpool2d":
                self.downsampler = nn.Sequential(
                    LayerNorm2d(in_channels, eps=eps),
                    nn.MaxPool2d(kernel_size=scale, stride=scale),
                    ChannelModification(in_channels, out_channels),
                )
            else:
                raise ValueError(f'Unknown value "{mode}" for `mode`.')

    def forward(self, x: Tensor) -> Tensor:
        return self.downsampler(x)


class Upsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int,
        mode: Literal["bicubic", "bilinear", "convtranspose2d", "pixelshuffle"] = "pixelshuffle",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.mode = mode

        if scale == 1:
            self.upsampler = nn.Identity()
        else:
            if mode in {"bicubic", "bilinear"}:
                self.upsampler = nn.Sequential(
                    nn.Upsample(scale_factor=scale, mode=mode),
                    ChannelModification(in_channels, out_channels),
                )
            elif mode == "convtranspose2d":
                self.upsampler = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale, stride=scale),
                    nn.GELU(),
                )
            elif mode == "pixelshuffle":
                self.upsampler = nn.Sequential(
                    ChannelModification(in_channels, out_channels * scale**2),
                    nn.PixelShuffle(scale),
                )
            else:
                raise ValueError(f'Unknown value "{mode}" for `mode`.')

    def forward(self, x: Tensor) -> Tensor:
        return self.upsampler(x)


class Collector(nn.Module):
    def __init__(
        self,
        levels: Sequence[tuple[int, int]],
        level: int,
        attention: bool = False,
        downsampler: Literal["conv2d", "maxpool2d"] = "conv2d",
        upsampler: Literal["bicubic", "bilinear", "convtranspose2d", "pixelshuffle"] = "pixelshuffle",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.levels = levels
        self.level = level
        self.attention = attention
        self.downsampler = downsampler
        self.upsampler = upsampler
        self.eps = eps

        channels = levels[level][0]
        self.concat = Concatenation(dim=1)
        self.from_encoder = nn.ModuleList(
            [
                Downsampler(
                    levels[i][0],
                    channels,
                    2 ** (level - i),
                    mode=downsampler,
                )
                for i in range(level + 1)
            ]
        )
        self.from_decoder = nn.ModuleList(
            [
                Upsampler(
                    levels[i][0],
                    channels,
                    2 ** (i - level),
                    mode=upsampler,
                )
                for i in range(len(levels) - 1, level, -1)
            ]
        )
        if attention:
            self.conv = ChannelModification((level + 1) * channels, channels)
            self.attention_gate = AttentionGate(channels, eps=eps)
        self.collector = ChannelModification((len(levels) - level * attention) * channels, channels)

    def forward(self, encoded: list[Tensor], decoded: list[Tensor]) -> Tensor:
        assert len(encoded) == len(self.from_encoder)
        assert len(decoded) == len(self.from_decoder)

        encoded = [self.from_encoder[i](encoded[i]) for i in range(len(encoded))]
        decoded = [self.from_decoder[i](decoded[i]) for i in range(len(decoded))]
        if self.attention:
            gated = self.attention_gate(self.conv(self.concat(*encoded)), decoded[-1])
            out = self.collector(self.concat(gated, *decoded))
        else:
            out = self.collector(self.concat(*encoded, *decoded))

        return out


class CDASSR(nn.Module):
    def __init__(
        self,
        levels: Sequence[tuple[int, int]] = BASE,
        in_channels: int = 3,
        out_channels: int = 3,
        block: str | Type[nn.Module] = "ResNetBlockV2",
        downsampler: Literal["conv2d", "maxpool2d"] = "maxpool2d",
        upsampler: Literal["bicubic", "bilinear", "convtranspose2d", "pixelshuffle"] = "pixelshuffle",
        super_upsampler: Literal["bicubic", "scale_aware"] = "scale_aware",
        n_recurrent: int = 0,
        channel_attention: bool = False,
        scale_aware_adaptation: bool = False,
        attention_gate: bool = False,
        concat_orig_interp: bool = False,
        layer_norm: bool = False,
        stochastic_depth_prob: float = 0.1,
        init_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        assert len(levels) >= 2
        assert in_channels >= 1
        assert out_channels >= 1
        assert n_recurrent >= 0
        assert 0.0 <= stochastic_depth_prob <= 1.0

        super().__init__()
        self.levels = levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = getattr(bottlenecks, block) if isinstance(block, str) else block
        self.downsampler = downsampler
        self.upsampler = upsampler
        self.super_upsampler = super_upsampler
        self.n_recurrent = n_recurrent
        self.channel_attention = channel_attention
        self.scale_aware_adaptation = scale_aware_adaptation
        self.attention_gate = attention_gate
        self.concat_orig_interp = concat_orig_interp
        self.layer_norm = layer_norm
        self.stochastic_depth_prob = stochastic_depth_prob
        self.init_weights = init_weights
        self.reduction: int = kwargs.pop("reduction", 16)
        self.n_experts: int = kwargs.pop("n_experts", 4)
        self.eps: float = kwargs.pop("eps", 1e-6)
        self.kwargs = kwargs

        self.concat = Concatenation(dim=1)
        self.stem = self.__make_stem()
        self.encoder = self.__make_encoder()
        self.decoder = self.__make_decoder()
        if self.super_upsampler == "scale_aware":
            self.scale_aware_upsampler = LASU(
                levels[0][0],
                n_experts=self.n_experts,
                reduction=self.reduction,
                eps=self.eps,
            )
        self.output = self.__make_output()
        self.auxiliary = self.__make_auxiliary()

        if init_weights:
            self.initialize_weights()

    def forward(
        self,
        x: Tensor,
        size: int | tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        assert len(x.shape) == 4
        assert x.shape[1] == self.in_channels
        assert x.shape[2] % (1 << (len(self.levels) - 1)) == 0
        assert x.shape[3] % (1 << (len(self.levels) - 1)) == 0

        if isinstance(size, int):
            size = (size, size)

        scale_h, scale_w = size[0] / x.shape[2], size[1] / x.shape[3]

        bicubic = nn.Upsample(size=size, mode="bicubic")

        stem = self.stem(x)
        out = stem

        # Encode
        encoded: list[Tensor] = []
        stem = self.stem(x)
        F_0 = stem

        out = F_0
        encoded: list[Tensor] = []

        for idx, encoder in enumerate(self.encoder):
            for i, enc in enumerate(encoder):
                if idx != len(self.encoder) - 1 and i == len(encoder) - 1:
                    out = enc(out)
                else:
                    block_out = enc(out, scale_h, scale_w)
                    if idx == 0:
                        out = block_out + F_0
                    else:
                        out = block_out

                if (idx != len(self.encoder) - 1 and i == len(encoder) - 2) or \
                        (idx == len(self.encoder) - 1 and i == len(encoder) - 1):
                    encoded.append(out)

        # Decode
        decoded = [encoded.pop()]
        for decoder in self.decoder:
            for i, dec in enumerate(decoder):
                if i == 0:
                    out = dec(encoded, decoded)
                else:
                    out = dec(out)
                    decoded.append(out)
                    encoded.pop()

        out += stem

        auxiliary = self.auxiliary(out)

        out = self.scale_aware_upsampler(out, *size) if self.super_upsampler == "scale_aware" else bicubic(out)
        out = self.concat(bicubic(stem), out) if self.concat_orig_interp else out
        out = self.output(out)

        return out, auxiliary

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __make_stem(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.levels[0][0], kernel_size=3, padding="same"),
            LayerNorm2d(self.levels[0][0], eps=self.eps),
        )

    def __make_encoder(self) -> nn.ModuleList:
        encoder = nn.ModuleList()
        total_blocks = sum(lvl[1] for lvl in self.levels)
        block_id = 1
        for idx, (channels, n_blocks) in enumerate(self.levels):
            level = nn.ModuleList()
            for i in range(1, n_blocks + 1):
                level.append(
                    RecurrentAttentionBlock(
                        TwiResBlock,
                        channels,
                        n_recurrent=self.n_recurrent,
                        attention=self.channel_attention,
                        scale_aware=True if self.scale_aware_adaptation else False,
                        layer_norm=self.layer_norm,
                        stochastic_depth_prob=self.stochastic_depth_prob * block_id / total_blocks,
                        reduction=self.reduction,
                        n_experts=self.n_experts,
                        eps=self.eps,
                        **self.kwargs,
                    )
                )
                block_id += 1
            if idx != len(self.levels) - 1:
                level.append(Downsampler(channels, self.levels[idx + 1][0], 2, mode=self.downsampler))
            encoder.append(level)
        return encoder

    def __make_decoder(self) -> nn.ModuleList:
        decoder = nn.ModuleList()

        for idx, (channels, _) in enumerate(self.levels[-2::-1], start=2):
            level = nn.ModuleList(
                [
                    Collector(
                        self.levels,
                        len(self.levels) - idx,
                        attention=self.attention_gate,
                        downsampler=self.downsampler,
                        upsampler=self.upsampler,
                        eps=self.eps,
                    ),
                    RecurrentAttentionBlock(
                        self.block,
                        channels,
                        n_recurrent=self.n_recurrent,
                        attention=self.channel_attention,
                        layer_norm=self.layer_norm,
                        reduction=self.reduction,
                        eps=self.eps,
                        **self.kwargs,
                    ),
                ]
            )
            decoder.append(level)

        return decoder

    def __make_output(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                self.levels[0][0] * (1 + self.concat_orig_interp),
                self.levels[0][0],
                kernel_size=3,
                padding="same",
            ),
            nn.GELU(),
            nn.Conv2d(self.levels[0][0], self.out_channels, kernel_size=3, padding="same"),
        )

    def __make_auxiliary(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(self.levels[0][0], self.levels[0][0], kernel_size=3, padding="same"),
            nn.GELU(),
            nn.Conv2d(self.levels[0][0], self.out_channels, kernel_size=3, padding="same"),
        )
