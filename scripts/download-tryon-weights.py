"""Download and verify the pinned local try-on evaluation weights."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

VTON_REVISION = "7720683168567eb5a2a4c67f15116c6e29c83ded"
DWPOSE_REVISION = "548b5df25b84d9f4aac0611dfa1c2a7a12f15571"
HUMAN_PARSER_REVISION = "1f80c34dbab321c5730dda5c3fea279fd3e97498"
EXPECTED = {
    "model.safetensors": (1_943_668_048, "d6cd38286885bc29fa487ea9383f80ffeb95862e7747c630d42c5d3c05bdd35a"),
    "dwpose/yolox_l.onnx": (216_746_733, "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411"),
    "dwpose/dw-ll_ucoco_384.onnx": (134_399_116, "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    args = parser.parse_args()
    weights_dir = args.weights_dir.resolve()
    hf_home = args.hf_home.resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    hf_hub_download(
        repo_id="fashn-ai/fashn-vton-1.5",
        filename="model.safetensors",
        revision=VTON_REVISION,
        local_dir=weights_dir,
        cache_dir=hf_home / "hub",
    )
    for filename in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
        hf_hub_download(
            repo_id="fashn-ai/DWPose",
            filename=filename,
            revision=DWPOSE_REVISION,
            local_dir=weights_dir / "dwpose",
            cache_dir=hf_home / "hub",
        )
    snapshot_download(
        repo_id="fashn-ai/fashn-human-parser",
        revision=HUMAN_PARSER_REVISION,
        allow_patterns=["config.json", "model.safetensors"],
        cache_dir=hf_home / "hub",
    )

    for relative, (expected_size, expected_hash) in EXPECTED.items():
        path = weights_dir / relative
        if not path.is_file() or path.stat().st_size != expected_size or sha256(path) != expected_hash:
            raise RuntimeError(f"weight verification failed: {relative}")
        print(f"verified {relative} ({expected_size} bytes)")
    print(f"human parser revision: {HUMAN_PARSER_REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
