"""Local JSONL engine for lightweight image checks and future model adapters."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

PROTOCOL_VERSION = 1
TRYON_FORMATS = {"JPEG", "PNG", "WEBP"}
TRYON_MAX_BYTES = 15 * 1024 * 1024
TRYON_MAX_PIXELS = 24_000_000
ALBUM_CLASSIFIER_REVISION = "mobilenet-v3-small-imagenet1k-v1"
ALBUM_CLASSIFIER_MAX_ITEMS = 256
ALBUM_CLASSIFIER_THRESHOLD = 0.35

# ImageNet contains many labels that do not belong in a private photo organizer.
# Only these coarse, non-sensitive groups can leave the uncategorized state.
OUTFIT_INDICES = {
    399, 400, 411, 414, 433, 439, 443, 445, 451, 452, 457, 459, 461, 464, 465,
    474, 490, 496, 501, 502, 514, 515, 518, 560, 568, 578, 601, 608, 610, 614,
    615, 617, 630, 638, 639, 643, 652, 655, 658, 679, 689, 697, 701, 715, 735,
    748, 770, 774, 775, 793, 796, 806, 808, 824, 834, 836, 837, 841, 842, 869,
    879, 887, 903, 906, 911,
}
TRANSPORT_INDICES = {
    403, 404, 405, 407, 408, 436, 444, 450, 466, 468, 472, 484, 510, 511, 517,
    537, 547, 554, 555, 561, 565, 569, 573, 575, 576, 586, 595, 603, 609, 612,
    625, 627, 628, 654, 656, 661, 665, 670, 671, 675, 690, 694, 705, 717, 734,
    751, 757, 779, 780, 802, 803, 812, 814, 817, 820, 829, 833, 847, 856, 864,
    866, 867, 870, 871, 874, 880, 895, 913, 914,
}
ARCHITECTURE_INDICES = {
    421, 424, 425, 449, 460, 467, 483, 497, 498, 500, 525, 536, 538, 540, 562,
    580, 582, 624, 634, 649, 660, 663, 668, 682, 698, 706, 718, 727, 743, 762,
    789, 819, 821, 825, 832, 839, 853, 858, 860, 873, 888, 900, 915,
}
ART_INDICES = {
    401, 402, 420, 432, 458, 486, 513, 541, 542, 546, 558, 566, 577, 579, 593,
    594, 611, 641, 642, 683, 684, 687, 696, 699, 739, 745, 749, 776, 822, 854,
    863, 875, 881, 883, 889, 917, 921,
}


def check_item(check_id: str, label: str, level: str, message: str, value: Any = None) -> dict[str, Any]:
    return {"id": check_id, "label": label, "level": level, "message": message, "value": value}


def person_presence_check(face_count: int, body_count: int) -> dict[str, Any]:
    likely_count = max(face_count, body_count)
    if likely_count > 1:
        return check_item("person_count", "人物数量", "block", "检测到多个人物候选，首版只支持单人照片", likely_count)
    if likely_count == 1:
        return check_item("person_count", "人物数量", "pass", "检测到一个人物候选", 1)
    return check_item("person_count", "人物数量", "warning", "轻量检测未确认人物，请人工确认照片中只有一人", 0)


def local_person_gate(image: Image.Image) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return [check_item("person_count", "人物数量", "warning", "本机未安装 OpenCV，暂时需要人工确认单人输入")], {"detector": "unavailable"}

    probe = image.copy()
    probe.thumbnail((640, 640), Image.Resampling.LANCZOS)
    rgb = np.asarray(probe.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    bodies, scores = hog.detectMultiScale(rgb, winStride=(8, 8), padding=(8, 8), scale=1.08)
    body_boxes = []
    if len(bodies):
        boxes = [[int(x), int(y), int(w), int(h)] for x, y, w, h in bodies]
        confidences = [float(value) for value in scores]
        kept = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.2, nms_threshold=0.35)
        body_boxes = [boxes[int(index)] for index in kept]

    checks = [person_presence_check(len(faces), len(body_boxes))]
    if len(body_boxes) == 1:
        x, y, width, height = body_boxes[0]
        height_ratio = height / max(1, probe.height)
        touches_edge = x <= 3 or y <= 3 or x + width >= probe.width - 3 or y + height >= probe.height - 3
        if height_ratio < 0.45:
            checks.append(check_item("person_scale", "人物占比", "warning", "人物在画面中较小，建议裁剪后再试穿", round(height_ratio, 3)))
        elif touches_edge:
            checks.append(check_item("person_scale", "人物边界", "warning", "人物贴近画面边缘，请确认头部到髋部完整", round(height_ratio, 3)))
        else:
            checks.append(check_item("person_scale", "人物占比", "pass", "人物在画面中的占比可用于下一步检查", round(height_ratio, 3)))
    return checks, {
        "detector": "opencv-haar-hog-v1",
        "face_candidates": len(faces),
        "body_candidates": len(body_boxes),
    }


def image_preflight(payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(payload.get("path") or ""))
    role = str(payload.get("role") or "")
    if role not in {"person", "garment"}:
        raise ValueError("role must be 'person' or 'garment'")
    if not path.is_file():
        raise ValueError("image file does not exist")

    file_size = path.stat().st_size
    checks: list[dict[str, Any]] = []
    checks.append(check_item(
        "file_size",
        "文件大小",
        "block" if file_size > TRYON_MAX_BYTES else "pass",
        "超过虚拟穿搭单图 15MB 上限" if file_size > TRYON_MAX_BYTES else "文件大小可用于预检",
        file_size,
    ))

    try:
        with Image.open(path) as source:
            image_format = str(source.format or "UNKNOWN").upper()
            width, height = source.size
            frame_count = int(getattr(source, "n_frames", 1))
            source.verify()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            pixels = width * height
            if pixels <= TRYON_MAX_PIXELS:
                sample = image.copy()
            else:
                sample = None
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"image decode failed: {exc}") from exc

    checks.append(check_item(
        "format",
        "图片格式",
        "pass" if image_format in TRYON_FORMATS and frame_count == 1 else "block",
        "格式可用于首版穿搭" if image_format in TRYON_FORMATS and frame_count == 1 else "首版穿搭仅支持单帧 JPEG、PNG 和 WebP",
        image_format,
    ))

    short_side = min(width, height)
    pixels = width * height
    if pixels > TRYON_MAX_PIXELS:
        dimension_level = "block"
        dimension_message = "解码后超过 2400 万像素上限"
    elif short_side < 384:
        dimension_level = "block"
        dimension_message = "图片短边小于 384px，细节不足"
    elif short_side < 768:
        dimension_level = "warning"
        dimension_message = "图片短边低于建议的 768px，结果细节可能受限"
    else:
        dimension_level = "pass"
        dimension_message = "图片尺寸达到首版建议范围"
    checks.append(check_item("dimensions", "图片尺寸", dimension_level, dimension_message, {"width": width, "height": height, "pixels": pixels}))

    metrics: dict[str, Any] = {
        "width": width,
        "height": height,
        "megapixels": round(pixels / 1_000_000, 2),
        "file_size_bytes": file_size,
        "format": image_format,
    }

    if sample is not None:
        sample.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        grayscale = ImageOps.grayscale(sample)
        histogram = grayscale.histogram()
        total = max(1, sum(histogram))
        brightness = float(ImageStat.Stat(grayscale).mean[0])
        clipped_dark = sum(histogram[:12]) / total
        clipped_bright = sum(histogram[244:]) / total
        edges = grayscale.filter(ImageFilter.FIND_EDGES)
        edge_score = math.sqrt(max(0.0, float(ImageStat.Stat(edges).var[0])))

        exposure_warning = brightness < 45 or brightness > 215 or clipped_dark > 0.45 or clipped_bright > 0.45
        checks.append(check_item(
            "exposure",
            "曝光",
            "warning" if exposure_warning else "pass",
            "画面可能过暗、过亮或存在大面积死黑死白" if exposure_warning else "整体曝光处于可检查范围",
            round(brightness, 1),
        ))

        sharpness_warning = edge_score < 8.0
        checks.append(check_item(
            "sharpness",
            "清晰度",
            "warning" if sharpness_warning else "pass",
            "边缘信息较少，请放大确认主体是否模糊" if sharpness_warning else "检测到足够的边缘信息",
            round(edge_score, 1),
        ))
        metrics.update({
            "brightness": round(brightness, 1),
            "edge_score": round(edge_score, 1),
            "clipped_dark_ratio": round(clipped_dark, 4),
            "clipped_bright_ratio": round(clipped_bright, 4),
        })

    if role == "person" and width > height * 1.35:
        checks.append(check_item("composition", "人物构图", "warning", "人物图为横向画幅，请确认头部到髋部完整可见"))
    else:
        checks.append(check_item("composition", "构图范围", "pass", "画幅方向未触发明显风险提示"))

    if role == "person" and sample is not None:
        person_checks, person_metrics = local_person_gate(sample)
        checks.extend(person_checks)
        metrics["person_gate"] = person_metrics

    levels = {item["level"] for item in checks}
    status = "block" if "block" in levels else "warning" if "warning" in levels else "pass"
    return {
        "role": role,
        "status": status,
        "metrics": metrics,
        "checks": checks,
        "limitations": ["人物数量使用本机轻量候选检测；姿态关键点、遮挡和服装分割仍需后续专用模型确认。"],
    }


def safe_category_for_imagenet_index(index: int) -> str | None:
    if 0 <= index <= 397:
        return "animals"
    if index in OUTFIT_INDICES:
        return "outfits"
    if index in TRANSPORT_INDICES:
        return "transport"
    if index in ARCHITECTURE_INDICES:
        return "architecture"
    if index in ART_INDICES:
        return "art"
    if 924 <= index <= 969:
        return "food"
    if 970 <= index <= 980:
        return "nature"
    if 984 <= index <= 990:
        return "plants"
    return None


def album_classify_safe_batch(payload: dict[str, Any]) -> dict[str, Any]:
    items = payload.get("items")
    enabled_categories = payload.get("enabled_categories")
    model_path = Path(str(payload.get("model_path") or ""))
    if not isinstance(items, list) or not items or len(items) > ALBUM_CLASSIFIER_MAX_ITEMS:
        raise ValueError(f"items must contain 1-{ALBUM_CLASSIFIER_MAX_ITEMS} photos")
    if not isinstance(enabled_categories, list):
        raise ValueError("enabled_categories must be an array")
    enabled = {str(value) for value in enabled_categories}
    if not model_path.is_file():
        raise ValueError("local classifier weights are not installed")

    try:
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as exc:
        raise ValueError("PyTorch and torchvision are required for local classification") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=None)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    preprocess = weights.transforms()
    labels = weights.meta["categories"]
    results: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each item must be an object")
        photo_id = str(item.get("id") or "")
        path = Path(str(item.get("path") or ""))
        if not photo_id or not path.is_file():
            results.append({"id": photo_id, "category_id": None, "confidence": 0.0, "label": None, "error": "image_unavailable"})
            continue
        try:
            with Image.open(path) as source:
                tensor = preprocess(source.convert("RGB")).unsqueeze(0).to(device)
            with torch.inference_mode():
                probabilities = model(tensor).softmax(dim=1)[0]
            confidence, class_index = probabilities.max(dim=0)
            index = int(class_index.item())
            score = float(confidence.item())
            category_id = safe_category_for_imagenet_index(index)
            accepted = category_id in enabled and score >= ALBUM_CLASSIFIER_THRESHOLD
            results.append({
                "id": photo_id,
                "category_id": category_id if accepted else None,
                "confidence": round(score, 6),
                "label": labels[index],
                "error": None,
            })
        except (OSError, ValueError, RuntimeError, Image.DecompressionBombError):
            results.append({"id": photo_id, "category_id": None, "confidence": 0.0, "label": None, "error": "classification_failed"})

    return {
        "revision": ALBUM_CLASSIFIER_REVISION,
        "device": str(device),
        "threshold": ALBUM_CLASSIFIER_THRESHOLD,
        "results": results,
    }


def respond(request: dict[str, Any]) -> dict[str, Any]:
    request_id = str(request.get("request_id") or uuid.uuid4())
    operation = request.get("op")

    if request.get("protocol_version") != PROTOCOL_VERSION:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "success": False,
            "error": {"code": "protocol_mismatch", "message": "Unsupported protocol version"},
        }

    if operation == "health":
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "success": True,
            "payload": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "engine": "vision-analytics-engine",
            },
        }

    if operation == "image.preflight":
        try:
            payload = request.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            result = image_preflight(payload)
        except (OSError, ValueError) as exc:
            return {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "success": False,
                "error": {"code": "preflight_failed", "message": str(exc)},
            }
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "success": True,
            "payload": result,
        }

    if operation == "album.classify_safe_batch":
        try:
            payload = request.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            result = album_classify_safe_batch(payload)
        except (OSError, ValueError) as exc:
            return {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "success": False,
                "error": {"code": "album_classification_failed", "message": str(exc)},
            }
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "success": True,
            "payload": result,
        }

    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "success": False,
        "error": {"code": "unknown_operation", "message": f"Unknown operation: {operation}"},
    }


def run_once() -> int:
    line = sys.stdin.readline()
    if not line:
        return 2
    try:
        request = json.loads(line)
        response = respond(request)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": "invalid",
            "success": False,
            "error": {"code": "invalid_request", "message": str(exc)},
        }
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Process one JSONL request and exit")
    args = parser.parse_args()
    if args.once:
        return run_once()

    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = respond(request)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": "invalid",
                "success": False,
                "error": {"code": "invalid_request", "message": str(exc)},
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
