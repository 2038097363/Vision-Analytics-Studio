"""Deterministic CPU baseline for clear 2D nuclear fluorescence images."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

PROTOCOL_VERSION = 1
ALGORITHM_REVISION = "otsu-watershed-v1"


def validate_parameters(payload: dict[str, Any]) -> tuple[float, int, bool, float | None]:
    threshold_scale = float(payload.get("threshold_scale", 1.0))
    min_area = int(payload.get("min_area", 40))
    exclude_edges = bool(payload.get("exclude_edges", True))
    scale_um_per_px = payload.get("scale_um_per_px")
    scale = float(scale_um_per_px) if scale_um_per_px not in (None, "") else None
    if not 0.5 <= threshold_scale <= 1.5:
        raise ValueError("阈值系数必须在 0.5 到 1.5 之间")
    if not 4 <= min_area <= 100_000:
        raise ValueError("最小核面积必须在 4 到 100000 像素之间")
    if scale is not None and not 0 < scale <= 10_000:
        raise ValueError("像素尺度必须大于 0 且不超过 10000 μm/px")
    return threshold_scale, min_area, exclude_edges, scale


def inspect_image(path: Path) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise ValueError("实验图像不存在")
    with Image.open(path) as opened:
        if int(getattr(opened, "n_frames", 1)) != 1:
            raise ValueError("首版只支持单页二维图像，不能分析多页 TIFF 或 Z-stack")
        oriented = ImageOps.exif_transpose(opened)
        width, height = oriented.size
        if width > 4096 or height > 4096 or width * height > 16_777_216:
            raise ValueError("图像超过当前 4096×4096 二维分析上限")
        original_mode = oriented.mode
        array = np.asarray(oriented)
        preview = oriented.convert("RGB")
    if array.ndim == 3:
        gray = array[..., :3].max(axis=2)
        channels = min(array.shape[2], 3)
    elif array.ndim == 2:
        gray = array
        channels = 1
    else:
        raise ValueError("当前只支持二维灰度或 RGB 图像")
    if not np.issubdtype(gray.dtype, np.integer) and not np.issubdtype(gray.dtype, np.floating):
        raise ValueError("不支持当前像素数据类型")
    return preview, gray.astype(np.float32), {
        "width": width, "height": height, "channels": channels,
        "source_mode": original_mode, "source_dtype": str(array.dtype),
    }


def segment(payload: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(str(payload.get("image_path") or ""))
    output_path = Path(str(payload.get("output_path") or ""))
    threshold_scale, min_area, exclude_edges, scale = validate_parameters(payload)
    preview, gray, metadata = inspect_image(image_path)
    started = time.perf_counter()

    from scipy import ndimage as ndi
    from skimage import exposure, feature, filters, measure, morphology, segmentation

    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        raise ValueError("图像没有可测量的有限像素")
    low, high = np.percentile(finite, [1, 99.8])
    if high <= low:
        normalized = np.zeros_like(gray, dtype=np.float32)
    else:
        normalized = exposure.rescale_intensity(gray, in_range=(low, high), out_range=(0.0, 1.0)).astype(np.float32)
    smooth = filters.gaussian(normalized, sigma=1.1, preserve_range=True)
    if float(smooth.max()) <= float(smooth.min()):
        binary = np.zeros_like(smooth, dtype=bool)
        base_threshold = 1.0
    else:
        base_threshold = float(filters.threshold_otsu(smooth))
        binary = smooth > min(base_threshold * threshold_scale, 0.9999)
    binary = morphology.remove_small_objects(binary, min_size=min_area)
    binary = morphology.binary_closing(binary, morphology.disk(1))

    distance = ndi.distance_transform_edt(binary)
    coordinates = feature.peak_local_max(distance, labels=binary, min_distance=4, exclude_border=False)
    marker_mask = np.zeros_like(binary, dtype=bool)
    if coordinates.size:
        marker_mask[tuple(coordinates.T)] = True
    markers, _ = ndi.label(marker_mask)
    labels = segmentation.watershed(-distance, markers, mask=binary) if markers.max() else np.zeros_like(binary, dtype=np.int32)
    for region in measure.regionprops(labels):
        if region.area < min_area:
            labels[labels == region.label] = 0
    labels, _, _ = segmentation.relabel_sequential(labels)

    measurements = []
    height, width = labels.shape
    for region in measure.regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        touches_edge = min_row == 0 or min_col == 0 or max_row == height or max_col == width
        if exclude_edges and touches_edge:
            continue
        area_px = int(region.area)
        area = area_px * scale * scale if scale is not None else float(area_px)
        diameter = 2.0 * math.sqrt(area / math.pi)
        measurements.append({
            "object_id": int(region.label), "area_pixels": area_px,
            "area": round(area, 4), "equivalent_diameter": round(diameter, 4),
            "centroid_x": round(float(region.centroid[1]), 2),
            "centroid_y": round(float(region.centroid[0]), 2), "touches_edge": touches_edge,
        })
    if len(measurements) > 5000:
        raise ValueError("检测对象超过 5000 个，请提高阈值或最小面积")

    valid_ids = {item["object_id"] for item in measurements}
    display_labels = labels.copy()
    if exclude_edges:
        display_labels[~np.isin(display_labels, list(valid_ids))] = 0
    boundary = segmentation.find_boundaries(display_labels, mode="outer")
    overlay = np.asarray(preview.resize((width, height))).copy()
    overlay[boundary] = np.array([214, 68, 52], dtype=np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, "RGB").save(output_path, format="PNG")

    areas = [item["area"] for item in measurements]
    diameters = [item["equivalent_diameter"] for item in measurements]
    unit = "μm" if scale is not None else "px"
    return {
        "schema_version": 1, "algorithm_revision": ALGORITHM_REVISION,
        "object_type": "nucleus", "review_status": "unreviewed",
        "metadata": metadata, "parameters": {
            "threshold_scale": threshold_scale, "otsu_threshold": round(base_threshold, 5),
            "min_area": min_area, "exclude_edges": exclude_edges,
            "scale_um_per_px": scale,
        },
        "summary": {
            "count": len(measurements), "unit": unit,
            "area_unit": f"{unit}²",
            "mean_area": round(float(np.mean(areas)), 4) if areas else None,
            "median_area": round(float(np.median(areas)), 4) if areas else None,
            "mean_equivalent_diameter": round(float(np.mean(diameters)), 4) if diameters else None,
        },
        "measurements": measurements,
        "overlay_path": str(output_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "当前结果是二维细胞核分割，核数量不能直接解释为细胞数量。",
            "自动分割尚未经过你的成像条件盲测，必须人工复核轮廓。",
        ],
    }


def respond(request: dict[str, Any]) -> dict[str, Any]:
    request_id = str(request.get("request_id") or "unknown")
    try:
        if request.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("不支持的协议版本")
        if request.get("op") != "lab.segment":
            raise ValueError("不支持的实验室引擎操作")
        result = segment(request.get("payload") or {})
        return {"protocol_version": 1, "request_id": request_id, "success": True, "payload": result, "error": None}
    except Exception as error:
        return {"protocol_version": 1, "request_id": request_id, "success": False, "payload": None, "error": {"code": "lab_failed", "message": str(error)}}


def main() -> None:
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(respond(json.loads(line)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
