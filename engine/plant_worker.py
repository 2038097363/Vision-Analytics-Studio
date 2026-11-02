"""Offline BioCLIP adapter for limited-scope plant candidate retrieval.

The scores returned here are similarities within a fixed product catalogue. They are
deliberately exposed as uncalibrated candidate scores, never as probabilities.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

PROTOCOL_VERSION = 1
MODEL_REVISION = "ce901ab3c6a913f9e9ef94ce6d27761069f4f01c"
CATALOG_REVISION = "cn-common-plants-v1-2026-09-19"
WEIGHTS_FILE = "open_clip_pytorch_model.bin"
EXPECTED_WEIGHTS_BYTES = 598_599_013
VALID_ORGANS = {"whole", "leaf", "flower", "fruit"}

PLANTS = [
    ("rosa-chinensis", "月季", "Rosa chinensis", "Chinese rose"),
    ("osmanthus-fragrans", "桂花", "Osmanthus fragrans", "sweet osmanthus"),
    ("camellia-japonica", "山茶", "Camellia japonica", "common camellia"),
    ("hibiscus-rosa-sinensis", "朱槿", "Hibiscus rosa-sinensis", "Chinese hibiscus"),
    ("bougainvillea-spectabilis", "三角梅", "Bougainvillea spectabilis", "great bougainvillea"),
    ("nelumbo-nucifera", "荷花", "Nelumbo nucifera", "sacred lotus"),
    ("chrysanthemum-morifolium", "菊花", "Chrysanthemum morifolium", "florist's chrysanthemum"),
    ("narcissus-tazetta", "水仙", "Narcissus tazetta", "paperwhite narcissus"),
    ("paeonia-suffruticosa", "牡丹", "Paeonia suffruticosa", "tree peony"),
    ("prunus-mume", "梅", "Prunus mume", "Chinese plum"),
    ("ginkgo-biloba", "银杏", "Ginkgo biloba", "ginkgo"),
    ("acer-palmatum", "鸡爪槭", "Acer palmatum", "Japanese maple"),
    ("magnolia-denudata", "玉兰", "Magnolia denudata", "lilytree"),
    ("jasminum-sambac", "茉莉", "Jasminum sambac", "Arabian jasmine"),
    ("gardenia-jasminoides", "栀子", "Gardenia jasminoides", "cape jasmine"),
    ("chlorophytum-comosum", "吊兰", "Chlorophytum comosum", "spider plant"),
    ("epipremnum-aureum", "绿萝", "Epipremnum aureum", "golden pothos"),
    ("monstera-deliciosa", "龟背竹", "Monstera deliciosa", "Swiss cheese plant"),
    ("dracaena-trifasciata", "虎尾兰", "Dracaena trifasciata", "snake plant"),
    ("aloe-vera", "芦荟", "Aloe vera", "aloe vera"),
]


def model_status(payload: dict[str, Any]) -> dict[str, Any]:
    weights_dir = Path(str(payload.get("weights_dir") or ""))
    weights = weights_dir / WEIGHTS_FILE
    size = weights.stat().st_size if weights.is_file() else None
    installed = size == EXPECTED_WEIGHTS_BYTES
    runtime_importable = (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("open_clip") is not None
    )
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    return {
        "installed": installed,
        "ready": installed and runtime_importable,
        "model_revision": MODEL_REVISION,
        "catalog_revision": CATALOG_REVISION,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "runtime_importable": runtime_importable,
        "weights_bytes": size or 0,
        "supported_taxa": len(PLANTS),
        "message": "BioCLIP 本地候选检索已就绪" if installed and runtime_importable else "植物模型运行时或权重不完整",
        "release_status": "experimental-uncalibrated-candidates",
    }


def candidate_decision(scores: list[float]) -> str:
    """Stay uncertain until the independent unknown-class validation is complete."""
    return "uncertain" if scores else "unsupported"


def identify(payload: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(str(payload.get("image_path") or ""))
    weights_dir = Path(str(payload.get("weights_dir") or ""))
    organ = str(payload.get("organ") or "whole")
    if organ not in VALID_ORGANS:
        raise ValueError("观察部位无效")
    if not image_path.is_file():
        raise ValueError("植物图片不存在")
    if not model_status({"weights_dir": str(weights_dir)})["ready"]:
        raise ValueError("BioCLIP 本地模型尚未就绪")

    started = time.perf_counter()
    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = weights_dir / WEIGHTS_FILE
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(weights_path)
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    labels = [f"{scientific}, {english}" for _, _, scientific, english in PLANTS]
    texts = tokenizer([f"a photo of {label}" for label in labels]).to(device)

    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        if width * height > 24_000_000:
            raise ValueError("图片超过当前 2400 万像素分析上限")
        tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        image_features = model.encode_image(tensor)
        text_features = model.encode_text(texts)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T)[0]
        display_scores = torch.softmax(similarities * 20.0, dim=-1)
        values, indices = display_scores.topk(3)

    candidates = []
    for value, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
        taxon_id, chinese_name, scientific_name, _ = PLANTS[index]
        candidates.append({
            "taxon_id": taxon_id,
            "chinese_name": chinese_name,
            "scientific_name": scientific_name,
            "score": round(float(value), 4),
        })
    return {
        "schema_version": 1,
        "decision": candidate_decision([item["score"] for item in candidates]),
        "candidates": candidates,
        "visible_features": [],
        "missing_evidence": ["候选分数尚未用独立未知类数据校准", "建议补充花、叶或果实特写后人工确认"],
        "observation_tags": [organ],
        "model_revision": MODEL_REVISION,
        "catalog_revision": CATALOG_REVISION,
        "device": device,
        "width": width,
        "height": height,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "notice": "以下仅为固定 20 种范围内的相似候选，不是确定识别结果。",
    }


def respond(request: dict[str, Any]) -> dict[str, Any]:
    request_id = str(request.get("request_id") or "unknown")
    try:
        if request.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("不支持的协议版本")
        operation = request.get("op")
        payload = request.get("payload") or {}
        if operation == "plant.status":
            result = model_status(payload)
        elif operation == "plant.identify":
            result = identify(payload)
        else:
            raise ValueError("不支持的植物引擎操作")
        return {"protocol_version": PROTOCOL_VERSION, "request_id": request_id, "success": True, "payload": result, "error": None}
    except Exception as error:  # JSONL boundary must always return structured failure.
        return {"protocol_version": PROTOCOL_VERSION, "request_id": request_id, "success": False, "payload": None, "error": {"code": "plant_failed", "message": str(error)}}


def main() -> None:
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(respond(json.loads(line)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
