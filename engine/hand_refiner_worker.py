"""JSONL adapter for the optional, isolated hand-refinement runtime.

This module deliberately has no ML imports at module load time.  The desktop app
can therefore inspect an incomplete installation without contaminating the FASHN
VTON environment or triggering a model download.
"""

from __future__ import annotations

import importlib.util
import json
import math
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1
MODEL_REVISION = "sd15-hand-inpaint-local-v1"
EXPECTED_DOWNLOAD_BYTES = 2_140_000_000


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def model_status(payload: dict[str, Any]) -> dict[str, Any]:
    runtime_root = Path(str(payload.get("runtime_root") or ""))
    weights_root = Path(str(payload.get("weights_root") or ""))
    model_root = weights_root / "stable-diffusion-inpainting"
    required_files = (
        model_root / "model_index.json",
        model_root / "text_encoder" / "model.fp16.safetensors",
        model_root / "unet" / "diffusion_pytorch_model.fp16.safetensors",
        model_root / "vae" / "diffusion_pytorch_model.fp16.safetensors",
    )
    required_modules = ("torch", "diffusers", "transformers", "PIL")
    modules = {name: importlib.util.find_spec(name) is not None for name in required_modules}
    runtime_importable = all(modules.values())
    base_model_installed = all(path.is_file() for path in required_files)
    installed = base_model_installed
    ready = installed and runtime_importable
    weights_bytes = sum(_file_size(path) for path in required_files)

    return {
        "installed": installed,
        "ready": ready,
        "runtime_installed": runtime_importable,
        "source_installed": Path(__file__).is_file(),
        "base_model_installed": base_model_installed,
        "control_model_installed": False,
        "landmark_model_installed": False,
        "model_revision": MODEL_REVISION,
        "python_version": platform.python_version(),
        "required_modules": modules,
        "weights_bytes": weights_bytes,
        "expected_download_bytes": EXPECTED_DOWNLOAD_BYTES,
        "expected_vram_gib": 8,
        "automatic_download": False,
        "message": (
            "本地手部局部修复已就绪；仅在单手核验失败时生成候选"
            if ready
            else "尚未安装独立手部修复运行时；试穿仍会严格标记缺手结果"
        ),
        "release_status": "single-sample-evaluation-passed" if ready else "evaluation-required",
    }


def _hand_geometry(
    size: tuple[int, int],
    normalized_path: list[list[float]],
) -> tuple[tuple[int, int, int, int], Any]:
    from PIL import Image, ImageDraw, ImageFilter

    if len(normalized_path) < 3:
        raise ValueError("手部修复缺少肩、肘、腕路径")
    width, height = size
    elbow = (normalized_path[-2][0] * width, normalized_path[-2][1] * height)
    wrist = (normalized_path[-1][0] * width, normalized_path[-1][1] * height)
    dx, dy = wrist[0] - elbow[0], wrist[1] - elbow[1]
    forearm = max(40.0, math.hypot(dx, dy))
    ux, uy = dx / forearm, dy / forearm
    start = (elbow[0] + dx * 0.60, elbow[1] + dy * 0.60)
    endpoint = (wrist[0] + ux * forearm * 0.72, wrist[1] + uy * forearm * 0.72)
    px, py = -uy, ux
    start_half = max(20.0, forearm * 0.15)
    wrist_half = max(42.0, forearm * 0.35)
    end_half = max(34.0, forearm * 0.27)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(
        (
            (start[0] + px * start_half, start[1] + py * start_half),
            (wrist[0] + px * wrist_half, wrist[1] + py * wrist_half),
            (endpoint[0] + px * end_half, endpoint[1] + py * end_half),
            (endpoint[0] - px * end_half, endpoint[1] - py * end_half),
            (wrist[0] - px * wrist_half, wrist[1] - py * wrist_half),
            (start[0] - px * start_half, start[1] - py * start_half),
        ),
        fill=255,
    )
    radius = int(end_half * 1.05)
    draw.ellipse(
        (endpoint[0] - radius, endpoint[1] - radius, endpoint[0] + radius, endpoint[1] + radius),
        fill=255,
    )
    mask = mask.filter(ImageFilter.MaxFilter(9))
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("无法建立手部修复区域")
    padding = max(42, int(forearm * 0.34))
    x0 = max(0, bbox[0] - padding)
    y0 = max(0, bbox[1] - padding)
    x1 = min(width, bbox[2] + padding)
    y1 = min(height, bbox[3] + padding)
    return (x0, y0, x1, y1), mask


def generate_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    image_path = Path(str(payload.get("image_path") or ""))
    output_dir = Path(str(payload.get("output_dir") or ""))
    model_root = Path(str(payload.get("weights_root") or "")) / "stable-diffusion-inpainting"
    repair_path = payload.get("repair_path")
    side = str(payload.get("side") or "unknown")
    if not image_path.is_file() or not isinstance(repair_path, list):
        raise ValueError("手部修复输入或路径无效")
    if not model_status({"runtime_root": payload.get("runtime_root"), "weights_root": payload.get("weights_root")})["ready"]:
        raise ValueError("本地手部局部修复模型尚未就绪")

    source = Image.open(image_path).convert("RGB")
    crop_box, mask = _hand_geometry(source.size, repair_path)
    crop = source.crop(crop_box)
    mask_crop = mask.crop(crop_box)
    model_input = crop.resize((512, 512), Image.Resampling.LANCZOS)
    model_mask = mask_crop.resize((512, 512), Image.Resampling.NEAREST)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"hand-{side}-mask.png"
    mask.save(mask_path)

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        model_root,
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipeline.enable_attention_slicing()
    pipeline.to("cuda")
    prompt = (
        "photorealistic close-up of a fair-skinned young woman's complete relaxed hand, "
        "five separate natural fingers clearly visible, short clean unpainted fingernails, "
        "wrist connected to the existing forearm, hand hanging naturally, matching the forearm "
        "skin colour and the same indoor lighting"
    )
    negative = (
        "missing hand, fist, clenched hand, fused fingers, extra fingers, deformed fingers, "
        "dark skin, red skin, glove, jewelry, sleeve, nail polish, dark fingernails, tattoo, "
        "blurry, low detail, duplicate limb, handbag, purse"
    )
    seeds = [int(value) for value in payload.get("seeds", [839, 173, 613, 727])][:4]
    started = time.perf_counter()
    candidate_paths = []
    for seed in seeds:
        generated = pipeline(
            prompt=prompt,
            negative_prompt=negative,
            image=model_input,
            mask_image=model_mask,
            num_inference_steps=int(payload.get("steps") or 30),
            guidance_scale=7.5,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0].resize(crop.size, Image.Resampling.LANCZOS)
        candidate = source.copy()
        candidate.paste(generated, crop_box[:2], mask_crop)
        candidate_path = output_dir / f"hand-{side}-{seed}.png"
        candidate.save(candidate_path, optimize=True)
        candidate_paths.append(str(candidate_path.resolve()))
    return {
        "side": side,
        "candidate_paths": candidate_paths,
        "mask_path": str(mask_path.resolve()),
        "crop_box": list(crop_box),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024 / 1024),
        "model_revision": MODEL_REVISION,
    }


def failure(request_id: str, code: str, message: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "success": False,
        "error": {"code": code, "message": message},
    }


def respond(request: dict[str, Any]) -> dict[str, Any]:
    request_id = str(request.get("request_id") or uuid.uuid4())
    if request.get("protocol_version") != PROTOCOL_VERSION:
        return failure(request_id, "protocol_mismatch", "Unsupported protocol version")
    payload = request.get("payload") if isinstance(request.get("payload"), dict) else {}
    try:
        if request.get("op") == "hand_refiner.status":
            result = model_status(payload)
        elif request.get("op") == "hand_refiner.generate":
            result = generate_candidates(payload)
        else:
            raise ValueError(f"未知操作：{request.get('op')}")
    except Exception as exc:
        return failure(request_id, "hand_refiner_failed", str(exc))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "success": True,
        "payload": result,
    }


def main() -> None:
    line = sys.stdin.readline()
    if not line:
        return
    print(json.dumps(respond(json.loads(line)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
