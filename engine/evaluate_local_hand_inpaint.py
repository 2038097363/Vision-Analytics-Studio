"""Generate isolated hand-repair candidates without touching the app pipeline.

This is an evaluation tool, not a production worker.  It deliberately requires
an explicit crop and mask polygon so a failed experiment cannot repaint clothing,
the face, a phone, or the already-valid hand.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def parse_box(value: str) -> tuple[int, int, int, int]:
    values = tuple(int(part.strip()) for part in value.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("box must be x0,y0,x1,y1")
    return values


def parse_polygon(value: str) -> list[tuple[int, int]]:
    points = []
    for item in value.split(";"):
        pair = tuple(int(part.strip()) for part in item.split(","))
        if len(pair) != 2:
            raise argparse.ArgumentTypeError("polygon must be x,y;x,y;...")
        points.append(pair)
    if len(points) < 3:
        raise argparse.ArgumentTypeError("polygon needs at least three points")
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--box", type=parse_box, required=True)
    parser.add_argument("--polygon", type=parse_polygon, required=True)
    parser.add_argument("--seeds", default="173,271,379,487")
    parser.add_argument("--steps", type=int, default=28)
    args = parser.parse_args()

    import torch
    from diffusers import StableDiffusionInpaintPipeline

    source = Image.open(args.image).convert("RGB")
    x0, y0, x1, y1 = args.box
    if not (0 <= x0 < x1 <= source.width and 0 <= y0 < y1 <= source.height):
        raise ValueError("crop is outside the image")
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).polygon(args.polygon, fill=255)
    crop = source.crop(args.box)
    mask_crop = mask.crop(args.box)
    model_input = crop.resize((512, 512), Image.Resampling.LANCZOS)
    model_mask = mask_crop.resize((512, 512), Image.Resampling.NEAREST)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_input.save(args.output_dir / "context.png")
    model_mask.save(args.output_dir / "mask.png")

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipeline.enable_attention_slicing()
    pipeline.to("cuda")
    prompt = (
        "photorealistic close-up of a fair-skinned young woman's complete relaxed left hand, "
        "five separate natural fingers clearly visible, short clean unpainted fingernails, "
        "wrist connected to the existing forearm, hand hanging naturally in front of a light blue "
        "fitted dress, matching the forearm skin colour and the same warm indoor lighting"
    )
    negative = (
        "missing hand, fist, clenched hand, fused fingers, extra fingers, deformed fingers, "
        "dark skin, red skin, black glove, jewelry, sleeve, nail polish, dark fingernails, tattoo, "
        "blurry, low detail, duplicate limb, handbag, purse"
    )
    started = time.perf_counter()
    for seed in (int(value) for value in args.seeds.split(",") if value.strip()):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        repaired_crop = pipeline(
            prompt=prompt,
            negative_prompt=negative,
            image=model_input,
            mask_image=model_mask,
            num_inference_steps=args.steps,
            guidance_scale=7.5,
            generator=generator,
        ).images[0].resize(crop.size, Image.Resampling.LANCZOS)
        feather = mask_crop.filter(ImageFilter.GaussianBlur(radius=2.0))
        result = source.copy()
        result.paste(Image.composite(repaired_crop, crop, feather), (x0, y0))
        result.save(args.output_dir / f"candidate-{seed}.png", optimize=True)
    print({
        "candidates": len([value for value in args.seeds.split(",") if value.strip()]),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "allocated_mib": round(torch.cuda.max_memory_allocated() / 1024 / 1024),
    })


if __name__ == "__main__":
    main()
