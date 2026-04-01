import argparse
from argparse import Namespace
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2

from model import CDASSR
from utils.pytorch_pipeline import PyTorchPipeline


def parse_args() -> tuple[Namespace, Callable[[], None]]:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("model_path", type=str, help="Path to CDASSR model (.pt).")
    parser.add_argument("img_path", type=str, help="Path to source image.")
    group.add_argument(
        "-size",
        "--size",
        nargs="+",
        type=int,
        help=(
            "Output size (width, height). "
            "If one value is provided, the output size will be (`size`, `size`). "
            "If multiple values are provided, the output size will be (`size[0]`, `size[1]`)."
        ),
    )
    group.add_argument(
        "-scale",
        "--scale",
        nargs="+",
        type=float,
        help=(
            "Factor by which the original size will be multiplied. "
            "If one value is provided, both sides will be multiplied by `scale`. "
            "If multiple values are provided, the original width will be multiplied by `scale[0]` "
            "while the original height will be multiplied by `scale[1]`."
        ),
    )

    return parser.parse_args(), parser.print_help


class Upscaler(object):
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_path = model_path
        self.device = PyTorchPipeline.get_device() if device is None else device

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = CDASSR(**checkpoint["configs"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        img_tensor: Optional[Tensor] = None,
        img_path: Optional[str] = None,
        size: Optional[int | tuple[int, int]] = None,
        scale: float | tuple[float, float] = 1.0,
    ) -> Tensor:
        img = img_path if img_tensor is None else img_tensor
        assert img is not None, "`img_tensor` and `img_path` cannot be both None."

        if isinstance(size, int):
            size = (size, size)
        if isinstance(scale, float):
            scale = (scale, scale)

        if isinstance(img, Tensor):
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
        elif isinstance(img, str):
            to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
            img: Tensor = to_tensor(Image.open(img).convert(mode="RGB")).unsqueeze(0)
        else:
            raise TypeError("Inappropriate type for either `img_tensor` or `img_path`, whichever is passed.")

        h, w = img.shape[-2:]

        if size is None:
            out_h, out_w = round(h * scale[0]), round(w * scale[1])
        else:
            out_h, out_w = size

        multiple = 1 << (len(self.model.levels) - 1)
        pad_h = multiple - rh if (rh := h % multiple) != 0 else 0
        pad_w = multiple - rw if (rw := w % multiple) != 0 else 0
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect").to(self.device)
        pred, _ = self.model(img, size=(out_h, out_w))
        pred = torch.clamp(pred, min=0.0, max=1.0)

        return pred[:, :, :out_h, :out_w]


if __name__ == "__main__":
    args, print_help = parse_args()

    upscaler = Upscaler(args.model_path)

    if args.size is not None:
        size = args.size[0] if len(args.size) == 1 else tuple(args.size[1::-1])
        upscaled = upscaler(img_path=args.img_path, size=size)
    elif args.scale is not None:
        scale = args.scale[0] if len(args.scale) == 1 else tuple(args.scale[1::-1])
        upscaled = upscaler(img_path=args.img_path, scale=scale)
    else:
        print_help()

    upscaled: Image.Image = v2.ToPILImage()(upscaled.squeeze())
    upscaled.save(f"upscaled_{PyTorchPipeline.get_datetime()}.png")
