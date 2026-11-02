"""Clean and verify one local inpainting candidate against the VTON quality gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.tryon_worker import (
    assess_limb_completeness,
    inspect_tryon_inputs,
    match_limb_skin_tone,
    preserve_hand_foreground,
    verify_garment_integrity,
)


def parse_polygon(value: str) -> list[tuple[int, int]]:
    return [tuple(int(part.strip()) for part in item.split(",")) for item in value.split(";")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", type=Path, required=True)
    parser.add_argument("--garment", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--human-parser", type=Path, required=True)
    parser.add_argument("--polygon", type=parse_polygon, required=True)
    parser.add_argument("--category", default="one-pieces")
    args = parser.parse_args()

    from fashn_human_parser import FashnHumanParser, LABELS_TO_IDS

    original_parser_init = FashnHumanParser.__init__

    def pinned_parser_init(self, model_id="", device=None):
        original_parser_init(self, model_id=str(args.human_parser), device=device)

    FashnHumanParser.__init__ = pinned_parser_init
    from fashn_vton import TryOnPipeline

    person = Image.open(args.person).convert("RGB")
    garment = Image.open(args.garment).convert("RGB")
    base = Image.open(args.base).convert("RGB")
    candidate = Image.open(args.candidate).convert("RGB")
    pipeline = TryOnPipeline(weights_dir=str(args.weights), device="cuda")
    analysis = inspect_tryon_inputs(pipeline, person, garment, args.category, "automatic")
    base_segmentation = np.asarray(pipeline.hp_model.predict(np.asarray(base)))
    candidate_segmentation = np.asarray(pipeline.hp_model.predict(np.asarray(candidate)))

    region_image = Image.new("L", base.size, 0)
    ImageDraw.Draw(region_image).polygon(args.polygon, fill=255)
    region = np.asarray(region_image) > 0
    skin_labels = (LABELS_TO_IDS["arms"], LABELS_TO_IDS["hands"])
    base_skin = np.isin(base_segmentation, skin_labels)
    candidate_skin = np.isin(candidate_segmentation, skin_labels)
    # The inpainting model can draw blue guide-like texture around an otherwise
    # valid hand.  Human parsing may call that texture an arm, so require the new
    # pixels to also resemble the already-correct forearm colour.
    import cv2

    base_lab = cv2.cvtColor(np.asarray(base), cv2.COLOR_RGB2LAB).astype(np.float32)
    candidate_lab = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2LAB).astype(np.float32)
    colour_reference = base_skin & ~region
    if int(colour_reference.sum()) >= 64:
        reference_lab = np.median(base_lab[colour_reference], axis=0)
        candidate_skin &= np.linalg.norm(candidate_lab - reference_lab, axis=2) <= 28.0
    candidate_skin &= region
    # First remove the malformed base hand with classical inpainting. Then copy
    # only parser- and colour-confirmed skin from the candidate. Replacing all
    # non-skin candidate pixels with the cleared base before feathering prevents
    # blue guide texture from bleeding into the dress at the hand boundary.
    old_hand = Image.fromarray(((base_skin & region).astype("uint8") * 255), mode="L").filter(
        ImageFilter.MaxFilter(5)
    )
    cleared_array = cv2.inpaint(
        cv2.cvtColor(np.asarray(base), cv2.COLOR_RGB2BGR),
        np.asarray(old_hand),
        5,
        cv2.INPAINT_TELEA,
    )
    cleared = cv2.cvtColor(cleared_array, cv2.COLOR_BGR2RGB)
    candidate_array = np.asarray(candidate)
    safe_candidate = np.where(candidate_skin[..., None], candidate_array, cleared)
    alpha = Image.fromarray((candidate_skin.astype("uint8") * 255), mode="L").filter(
        ImageFilter.GaussianBlur(0.7)
    )
    cleaned = Image.composite(Image.fromarray(safe_candidate), Image.fromarray(cleared), alpha)

    cleaned_segmentation = pipeline.hp_model.predict(np.asarray(cleaned))
    cleaned, before_delta, after_delta, reference_modes = match_limb_skin_tone(
        person,
        cleaned,
        analysis["person_segmentation"],
        cleaned_segmentation,
        LABELS_TO_IDS,
        analysis["pose"],
    )
    # Final local hand correction uses the already-correct generated forearm as
    # the closest lighting reference.  This is stricter than a global face or
    # opposite-arm match and avoids a pink/dark hand at the wrist seam.
    tone_segmentation = np.asarray(pipeline.hp_model.predict(np.asarray(cleaned)))
    target_skin = candidate_skin & np.isin(tone_segmentation, skin_labels)
    yy, xx = np.indices(region.shape)
    reference_skin = base_skin & ~region & (xx < int(base.width * 0.62)) & (yy > int(base.height * 0.28))
    if int(target_skin.sum()) >= 64 and int(reference_skin.sum()) >= 64:
        cleaned_rgb = np.asarray(cleaned).copy()
        cleaned_lab = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        shift = np.median(base_lab[reference_skin], axis=0) - np.median(cleaned_lab[target_skin], axis=0)
        shift = np.clip(shift, (-36.0, -10.0, -10.0), (36.0, 10.0, 10.0))
        shifted_lab = cleaned_lab.copy()
        shifted_lab[target_skin] = np.clip(shifted_lab[target_skin] + shift, 0, 255)
        shifted_rgb = cv2.cvtColor(shifted_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        tone_alpha = Image.fromarray((target_skin.astype("uint8") * 255), mode="L").filter(
            ImageFilter.GaussianBlur(0.6)
        )
        cleaned = Image.composite(Image.fromarray(shifted_rgb), cleaned, tone_alpha)
    cleaned, phone_preserved = preserve_hand_foreground(
        person,
        cleaned,
        analysis["person_segmentation"],
        LABELS_TO_IDS["hands"],
        LABELS_TO_IDS["bag"],
        analysis["pose"],
    )
    final_segmentation = pipeline.hp_model.predict(np.asarray(cleaned))
    final_pose = pipeline.pose_model(np.asarray(cleaned)[..., ::-1])
    limbs = assess_limb_completeness(
        final_segmentation,
        analysis["pose"],
        analysis["category"],
        analysis["garment_ratios"],
        LABELS_TO_IDS,
        final_pose,
        cleaned,
        analysis["person_segmentation"],
    )
    garment_ok, garment_message = verify_garment_integrity(
        final_segmentation,
        analysis["category"],
        LABELS_TO_IDS,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.save(args.output, optimize=True)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "garment": {"passed": garment_ok, "message": garment_message},
        "limbs": limbs,
        "skin_tone": {
            "delta_before": before_delta,
            "delta_after": after_delta,
            "reference_modes": reference_modes,
        },
        "phone_preserved": phone_preserved,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
