"""Isolated JSONL adapter for the optional FASHN VTON evaluation runtime."""

from __future__ import annotations

import json
import importlib.metadata
import importlib.util
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

PROTOCOL_VERSION = 1
MODEL_REVISION = "7720683168567eb5a2a4c67f15116c6e29c83ded"
SOURCE_REVISION = "7c0f10af3f91ad4048fe9729c470a13ef905d25a"
HUMAN_PARSER_REVISION = "1f80c34dbab321c5730dda5c3fea279fd3e97498"
EXPECTED_FILES = {
    "model.safetensors": 1_943_668_048,
    "dwpose/yolox_l.onnx": 216_746_733,
    "dwpose/dw-ll_ucoco_384.onnx": 134_399_116,
}
VALID_CATEGORIES = {"automatic", "tops", "bottoms", "one-pieces"}
VALID_GARMENT_TYPES = {"model", "flat-lay"}
VALID_SLEEVE_MODES = {"automatic", "standard", "reveal-arms"}


def preserve_hand_foreground(
    person_image: Image.Image,
    generated_image: Image.Image,
    segmentation: Any,
    hands_label: int,
    bag_label: int,
    pose: Any = None,
) -> tuple[Image.Image, bool]:
    """Put parser-confirmed hands and held objects back over the generated garment.

    Diffusion models can redraw fingers even when the clothing-agnostic input keeps
    them. The parser's hands label gives us a conservative, local-only matte.
    ``bag`` is included because phones and small handheld items are often assigned
    that class. DWPose supplements it with a compact matte around an upper hand,
    which reliably protects a phone in a mirror selfie even when the parser calls
    the phone background. A free, lower hand is left for the VTON model to finish.
    """
    import numpy as np
    from PIL import ImageDraw, ImageFilter

    target_size = generated_image.size
    source = person_image.convert("RGB")
    if source.size != target_size:
        source = source.resize(target_size, Image.Resampling.LANCZOS)

    segmentation_array = np.asarray(segmentation)
    if segmentation_array.shape[:2] != (target_size[1], target_size[0]):
        segmentation_array = np.asarray(
            Image.fromarray(segmentation_array.astype("uint8")).resize(
                target_size,
                Image.Resampling.NEAREST,
            )
        )

    hands = segmentation_array == hands_label
    held_items = segmentation_array == bag_label

    # A phone-holding hand is usually adjacent to the parser's ``bag`` region.
    # Do not overlay every hand: doing so would reintroduce the original long
    # sleeve at a free wrist that the VTON model should complete.
    protected = held_items.copy()
    if bool(held_items.any()):
        held_item_matte = Image.fromarray((held_items.astype("uint8") * 255), mode="L")
        held_item_neighborhood = np.asarray(held_item_matte.filter(ImageFilter.MaxFilter(17))) > 0
        protected |= hands & held_item_neighborhood

    # DWPose reports 21 keypoints for each visible hand. Preserve every parser-
    # confirmed source hand inside its own keypoint neighbourhood. This keeps a
    # free lower hand as well as a phone-holding hand, without copying a sleeve.
    detected_hands = pose.get("hands") if isinstance(pose, dict) else None
    if getattr(detected_hands, "ndim", 0) == 3:
        image_height, image_width = protected.shape
        pose_matte = Image.new("L", (image_width, image_height), 0)
        pose_draw = ImageDraw.Draw(pose_matte)
        hand_connections = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        )
        for hand in detected_hands:
            points = hand[(hand[:, 0] >= 0) & (hand[:, 1] >= 0)]
            if len(points) < 12:
                continue
            support_radius = max(3, int(min(image_width, image_height) * 0.008))
            supported_points = 0
            for point_x, point_y in points:
                px, py = int(point_x * image_width), int(point_y * image_height)
                y0, y1 = max(0, py - support_radius), min(image_height, py + support_radius + 1)
                x0, x1 = max(0, px - support_radius), min(image_width, px + support_radius + 1)
                supported_points += bool(hands[y0:y1, x0:x1].any())
            # Pose estimators can hallucinate a full hand over a pocket or sleeve.
            # Require the source parser to support at least half of its landmarks.
            if supported_points < 11:
                continue
            x_min = float(points[:, 0].min())
            x_max = float(points[:, 0].max())
            y_min = float(points[:, 1].min())
            y_max = float(points[:, 1].max())
            center_x = float(points[:, 0].mean())
            is_upper_hand = float(points[:, 1].mean()) < 0.58
            if is_upper_hand:
                # The phone normally sits between the upper hand and the face.
                if center_x >= 0.5:
                    x0, x1 = x_min - 0.006, x_max - 0.009
                else:
                    x0, x1 = x_min + 0.009, x_max + 0.006
                y0 = y_min - 0.078
                phone_width = max(1, int((x1 - x0) * image_width))
                y1_pixels = min(
                    int((y_max - 0.020) * image_height),
                    int(y0 * image_height) + int(phone_width * 2.05),
                )
                rectangle = (
                    max(0, int(x0 * image_width)),
                    max(0, int(y0 * image_height)),
                    min(image_width, int(x1 * image_width)),
                    min(image_height, y1_pixels),
                )
                # GrabCut keeps the phone silhouette and removes background that
                # happens to fall inside its bounding rectangle, such as a mirror
                # or the original shirt. It is classical local CV, so it adds no
                # cloud request or model download.
                try:
                    import cv2

                    margin = max(3, int(image_width * 0.012))
                    grab_x0 = max(0, rectangle[0] - margin)
                    grab_y0 = max(0, rectangle[1] - margin)
                    grab_x1 = min(image_width, rectangle[2] + margin)
                    grab_y1 = min(image_height, rectangle[3] + margin)
                    grabcut_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                    background_model = np.zeros((1, 65), dtype=np.float64)
                    foreground_model = np.zeros((1, 65), dtype=np.float64)
                    cv2.grabCut(
                        np.asarray(source).copy(),
                        grabcut_mask,
                        (grab_x0, grab_y0, grab_x1 - grab_x0, grab_y1 - grab_y0),
                        background_model,
                        foreground_model,
                        4,
                        cv2.GC_INIT_WITH_RECT,
                    )
                    phone_mask = (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)
                    bounded_phone = np.zeros_like(phone_mask)
                    bounded_phone[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = True
                    phone_mask &= bounded_phone
                    # Keep dark camera lenses and small stickers enclosed by the
                    # recognised phone silhouette instead of turning them into
                    # holes from the generated image.
                    filled = (phone_mask.astype("uint8") * 255)
                    flood = filled.copy()
                    cv2.floodFill(flood, np.zeros((image_height + 2, image_width + 2), np.uint8), (0, 0), 255)
                    phone_mask |= cv2.bitwise_not(flood).astype(bool)
                    protected |= phone_mask
                except Exception:
                    # A tight rectangle is still safer than allowing the phone to
                    # be regenerated when OpenCV cannot create a foreground mask.
                    protected[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = True
            stroke_width = max(3, int(image_width * 0.014))
            pixel_points = [(int(x * image_width), int(y * image_height)) for x, y in points]
            for start, end in hand_connections:
                pose_draw.line((pixel_points[start], pixel_points[end]), fill=255, width=stroke_width)
            for point in pixel_points:
                pose_draw.ellipse(
                    (point[0] - stroke_width // 2, point[1] - stroke_width // 2,
                     point[0] + stroke_width // 2, point[1] + stroke_width // 2),
                    fill=255,
                )
            # Copy only pixels that the source parser identifies as a hand. The
            # dilated keypoint skeleton selects the correct component and keeps
            # neighbouring original clothing out of the composite.
            hand_neighbourhood = np.asarray(pose_matte.filter(ImageFilter.MaxFilter(9))) > 0
            protected |= hands & hand_neighbourhood

        # The pose matte itself is used only for the upper holding hand where it
        # also covers small gaps between fingers and the phone edge.
        upper_pose = Image.new("L", (image_width, image_height), 0)
        upper_draw = ImageDraw.Draw(upper_pose)
        for hand in detected_hands:
            points = hand[(hand[:, 0] >= 0) & (hand[:, 1] >= 0)]
            if len(points) < 12 or float(points[:, 1].mean()) >= 0.58:
                continue
            support_radius = max(3, int(min(image_width, image_height) * 0.008))
            if sum(
                bool(hands[
                    max(0, int(y * image_height) - support_radius):min(image_height, int(y * image_height) + support_radius + 1),
                    max(0, int(x * image_width) - support_radius):min(image_width, int(x * image_width) + support_radius + 1),
                ].any())
                for x, y in points
            ) < 11:
                continue
            pixel_points = [(int(x * image_width), int(y * image_height)) for x, y in points]
            stroke_width = max(3, int(image_width * 0.014))
            for start, end in hand_connections:
                upper_draw.line((pixel_points[start], pixel_points[end]), fill=255, width=stroke_width)
        protected |= np.asarray(upper_pose.filter(ImageFilter.MaxFilter(7))) > 0

    if not bool(protected.any()):
        return generated_image, False

    # Keep the margin deliberately tight: it protects the wrist edge without
    # pasting a rectangular patch of the original clothing onto the result.
    matte = Image.fromarray((protected.astype("uint8") * 255), mode="L")
    matte = matte.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(0.7))
    restored = Image.composite(source, generated_image.convert("RGB"), matte)
    return restored, True


def merge_targeted_limb_repairs(
    garment_base: Image.Image,
    reconstruction: Image.Image,
    reconstruction_segmentation: Any,
    repair_targets: list[str],
    pose: Any,
    labels_to_ids: dict[str, int],
    repair_paths: dict[str, list[list[float]]] | None = None,
) -> tuple[Image.Image, bool]:
    """Merge reconstructed skin only inside explicitly missing limb segments."""
    import numpy as np
    from PIL import ImageDraw, ImageFilter

    target_size = garment_base.size
    if not repair_targets:
        return garment_base.convert("RGB"), False
    reconstruction_image = reconstruction.convert("RGB")
    if reconstruction_image.size != target_size:
        reconstruction_image = reconstruction_image.resize(target_size, Image.Resampling.LANCZOS)

    segmentation = np.asarray(reconstruction_segmentation)
    if segmentation.shape[:2] != (target_size[1], target_size[0]):
        segmentation = np.asarray(
            Image.fromarray(segmentation.astype("uint8")).resize(target_size, Image.Resampling.NEAREST)
        )

    candidate = pose.get("bodies", {}).get("candidate") if isinstance(pose, dict) else None
    if getattr(candidate, "ndim", 0) != 2:
        return garment_base.convert("RGB"), False

    width, height = target_size
    region = Image.new("L", target_size, 0)
    draw = ImageDraw.Draw(region)
    chain_indices = {
        "arm_left": (2, 3, 4),
        "arm_right": (5, 6, 7),
        "leg_left": (8, 9, 10),
        "leg_right": (11, 12, 13),
    }
    endpoint_indices = {
        "hand_left": 4,
        "hand_right": 7,
        "foot_left": 10,
        "foot_right": 13,
    }
    for target in repair_targets:
        resolved_path = (repair_paths or {}).get(target)
        if target in chain_indices:
            if resolved_path:
                points = [
                    (int(float(point[0]) * width), int(float(point[1]) * height))
                    for point in resolved_path
                ]
            else:
                indices = chain_indices[target]
                points = [
                    (int(float(candidate[index, 0]) * width), int(float(candidate[index, 1]) * height))
                    for index in indices
                ]
            line_width = max(8, int(min(width, height) * (0.075 if target.startswith("arm") else 0.11)))
            draw.line(points, fill=255, width=line_width, joint="curve")
        elif target in endpoint_indices:
            if resolved_path:
                point = resolved_path[-1]
            else:
                point = candidate[endpoint_indices[target]]
            x, y = int(float(point[0]) * width), int(float(point[1]) * height)
            radius = max(8, int(min(width, height) * (0.065 if target.startswith("hand") else 0.08)))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    repair_region = np.asarray(region) > 0
    reconstructed_skin = np.isin(
        segmentation,
        (labels_to_ids["arms"], labels_to_ids["hands"], labels_to_ids["legs"], labels_to_ids["feet"]),
    )
    limb_mask = repair_region & reconstructed_skin
    if not bool(limb_mask.any()):
        return garment_base.convert("RGB"), False

    # Keep the repair inside parser-confirmed skin, with a narrow soft boundary.
    matte = Image.fromarray((limb_mask.astype("uint8") * 255), mode="L")
    matte = matte.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8))
    return Image.composite(reconstruction_image, garment_base.convert("RGB"), matte), True


def choose_automatic_strategy(
    category: str,
    person: dict[str, float],
    garment: dict[str, float],
    pose_visibility: float,
) -> tuple[str, str]:
    """Choose preservation or reconstruction from local parser and pose evidence.

    This is deliberately conservative: a failed reconstruction is more visible
    than retaining a covered area. Users can still select the manual force mode.
    Ratios are label area / full image area from FASHN Human Parser.
    """
    person_upper = sum(person.get(label, 0.0) for label in ("top", "dress", "scarf", "torso"))
    person_lower = sum(person.get(label, 0.0) for label in ("skirt", "pants", "legs"))
    garment_arm_skin = garment.get("arms", 0.0)
    person_arm_skin = person.get("arms", 0.0)
    garment_leg_skin = garment.get("legs", 0.0)
    person_leg_skin = person.get("legs", 0.0)
    target_exposes_arms = garment_arm_skin >= 0.014
    source_arms_likely_covered = person_upper >= 0.05 and person_arm_skin < 0.012
    target_exposes_legs = garment_leg_skin >= 0.025
    source_legs_likely_covered = person_lower >= 0.08 and person_leg_skin < 0.018
    pose_is_usable = pose_visibility >= 0.18

    if category == "tops":
        if target_exposes_arms and source_arms_likely_covered and pose_is_usable:
            return (
                "reconstruct-coverage",
                "本地解析判断目标服装露出手臂、原图手臂被覆盖且姿态可用，因此补全整段手臂；手持手机处会单独保留。",
            )
        return "preserve-body", "未同时满足可靠的袖长差异与姿态条件，优先保持原身体区域稳定。"

    if category == "bottoms":
        if target_exposes_legs and source_legs_likely_covered and pose_is_usable:
            return (
                "reconstruct-coverage",
                "本地解析判断目标下装露出腿部、原图腿部被覆盖且姿态可用，因此补全需要露出的腿部；手持物区域会单独保留。",
            )
        return "preserve-body", "未同时满足可靠的裤长或裙长差异与姿态条件，优先保持原身体区域稳定。"

    if category == "one-pieces":
        if not pose_is_usable:
            return "preserve-body", "人物姿态关键点不足，优先保持原图，避免重建躯干和下半身。"
        exposed_parts = []
        if target_exposes_arms and source_arms_likely_covered:
            exposed_parts.append("手臂")
        if target_exposes_legs and source_legs_likely_covered:
            exposed_parts.append("腿部")
        reconstructed = "、".join(exposed_parts) if exposed_parts else "躯干与下半身"
        return "reconstruct-coverage", f"人物全身和姿态信息充分，按连衣裙范围重建{reconstructed}；手持手机处会单独保留。"

    return "preserve-body", "未识别到可处理的服装类别，优先保持原身体区域稳定。"


def choose_garment_category(garment: dict[str, float]) -> tuple[str, str]:
    """Classify the supplied garment from local human-parser label coverage."""
    upper = sum(garment.get(label, 0.0) for label in ("top", "dress", "scarf"))
    lower = sum(garment.get(label, 0.0) for label in ("skirt", "pants", "belt"))
    dress = garment.get("dress", 0.0)

    if dress >= 0.035 and dress >= garment.get("top", 0.0) * 0.45:
        return "one-pieces", "本地服装解析检测到连续裙装区域。"
    if lower >= 0.055 and lower > upper * 1.25:
        return "bottoms", "本地服装解析检测到以下装区域为主。"
    return "tops", "本地服装解析检测到以上装区域为主。"


def evaluate_tryon_scene(
    person: dict[str, float],
    garment: dict[str, float],
) -> dict[str, Any]:
    """Reject inputs the single-garment model cannot represent faithfully."""
    person_coverage = sum(
        person.get(label, 0.0)
        for label in (
            "face", "hair", "top", "dress", "skirt", "pants",
            "arms", "hands", "legs", "feet",
        )
    )
    target_top = garment.get("top", 0.0)
    target_one_piece = garment.get("dress", 0.0)
    # A model photo normally contains trousers or a skirt outside the requested
    # upper garment.  That is supported.  A sizeable top and dress region at the
    # same time indicates overlapping layers (for example a sweater under denim
    # overalls), which the single-category model cannot separate reliably.
    layered_reference = target_top >= 0.10 and target_one_piece >= 0.04
    blockers: list[str] = []
    if layered_reference:
        blockers.append(
            "目标图同时包含多件相互遮挡的服装；请分别上传单件上衣、下装或连衣裙"
        )
    if person_coverage < 0.12:
        blockers.append(
            "人物在画面中占比过小；请裁剪到人物至少占画面约四分之一，并保留需要换装的完整身体区域"
        )
    return {
        "supported": not blockers,
        "blockers": blockers,
        "person_coverage": round(person_coverage, 4),
        "layered_reference": layered_reference,
    }


def inspect_tryon_inputs(
    pipeline: Any,
    person_image: Image.Image,
    garment_image: Image.Image,
    requested_category: str,
    sleeve_mode: str,
    garment_photo_type: str = "model",
) -> dict[str, Any]:
    """Parse both inputs once and derive every downstream generation decision."""
    import numpy as np
    from fashn_human_parser import LABELS_TO_IDS

    person_np = np.array(person_image)
    garment_np = np.array(garment_image)
    person_segmentation = pipeline.hp_model.predict(person_np)
    garment_segmentation = pipeline.hp_model.predict(garment_np)
    pose = pipeline.pose_model(person_np[..., ::-1])

    def ratios(segmentation: Any) -> dict[str, float]:
        pixels = max(1, int(segmentation.size))
        return {
            label: float((segmentation == label_id).sum()) / pixels
            for label, label_id in LABELS_TO_IDS.items()
        }

    pose_subset = pose.get("bodies", {}).get("subset") if isinstance(pose, dict) else None
    pose_visibility = float((pose_subset >= 0).mean()) if getattr(pose_subset, "size", 0) else 0.0
    person_ratios = ratios(person_segmentation)
    garment_ratios = ratios(garment_segmentation)
    scene = evaluate_tryon_scene(person_ratios, garment_ratios)
    if garment_photo_type == "flat-lay" and scene["layered_reference"]:
        scene["layered_reference"] = False
        scene["blockers"] = [
            blocker for blocker in scene["blockers"]
            if not blocker.startswith("目标图同时包含多件")
        ]
        scene["supported"] = not scene["blockers"]
    if not scene["supported"]:
        raise ValueError("输入场景暂不支持：" + "；".join(scene["blockers"]))
    if requested_category == "automatic":
        category, category_notice = choose_garment_category(garment_ratios)
    else:
        category, category_notice = requested_category, "已使用手动指定的服装类别。"

    if sleeve_mode == "standard":
        adaptive_strategy, coverage_notice = "preserve-body", "已选择优先保留原身体区域。"
    elif sleeve_mode == "reveal-arms":
        adaptive_strategy, coverage_notice = (
            "reconstruct-coverage",
            "已手动选择补全遮挡区域；生成内容是模型推测，不是对真实细节的还原。",
        )
    else:
        adaptive_strategy, coverage_notice = choose_automatic_strategy(
            category,
            person_ratios,
            garment_ratios,
            pose_visibility,
        )

    return {
        "category": category,
        "category_notice": category_notice,
        "adaptive_strategy": adaptive_strategy,
        "coverage_notice": coverage_notice,
        "person_segmentation": person_segmentation,
        "garment_segmentation": garment_segmentation,
        "person_ratios": person_ratios,
        "garment_ratios": garment_ratios,
        "pose": pose,
        "pose_visibility": pose_visibility,
        "scene": scene,
    }


def analyze_tryon_scene(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the semantic input gate without loading the diffusion model."""
    import numpy as np
    from fashn_human_parser import FashnHumanParser, LABELS_TO_IDS

    person_path = Path(str(payload.get("person_path") or ""))
    garment_path = Path(str(payload.get("garment_path") or ""))
    human_parser_dir = Path(str(payload.get("human_parser_dir") or ""))
    garment_photo_type = str(payload.get("garment_photo_type") or "model")
    if not person_path.is_file() or not garment_path.is_file():
        raise ValueError("人物或服装素材不存在")
    parser = FashnHumanParser(model_id=str(human_parser_dir), device="cuda")
    with Image.open(person_path) as source:
        person_image = ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(garment_path) as source:
        garment_image = ImageOps.exif_transpose(source).convert("RGB")

    def ratios(image: Image.Image) -> dict[str, float]:
        segmentation = np.asarray(parser.predict(np.asarray(image)))
        pixels = max(1, int(segmentation.size))
        return {
            label: float((segmentation == label_id).sum()) / pixels
            for label, label_id in LABELS_TO_IDS.items()
        }

    person_ratios = ratios(person_image)
    garment_ratios = ratios(garment_image)
    scene = evaluate_tryon_scene(person_ratios, garment_ratios)
    if garment_photo_type == "flat-lay" and scene["layered_reference"]:
        # Human-parser labels overlap more on product-only flat lays.  The layer
        # blocker applies to model-worn references where garments occlude each other.
        scene["layered_reference"] = False
        scene["blockers"] = [
            blocker for blocker in scene["blockers"]
            if not blocker.startswith("目标图同时包含多件")
        ]
        scene["supported"] = not scene["blockers"]
    category, category_notice = choose_garment_category(garment_ratios)
    return {
        **scene,
        "category": category,
        "category_notice": category_notice,
        "person_ratios": person_ratios,
        "garment_ratios": garment_ratios,
    }


def assess_limb_completeness(
    segmentation: Any,
    pose: Any,
    category: str,
    garment_ratios: dict[str, float],
    labels_to_ids: dict[str, int],
    output_pose: Any | None = None,
    output_image: Image.Image | None = None,
    source_segmentation: Any | None = None,
) -> dict[str, Any]:
    """Check each expected limb segment and hand/foot instead of counting blobs."""
    import cv2
    import numpy as np

    segmentation_array = np.asarray(segmentation)
    image_area = max(1, int(segmentation_array.size))
    minimum_area = max(32, int(image_area * 0.004))

    def components(labels: tuple[int, ...]) -> int:
        mask = np.isin(segmentation_array, labels).astype("uint8")
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        return sum(
            int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area
            for index in range(1, count)
        )

    bodies = pose.get("bodies", {}) if isinstance(pose, dict) else {}
    subset = bodies.get("subset")
    candidate = bodies.get("candidate")
    subset_row = subset[0] if getattr(subset, "ndim", 0) == 2 and len(subset) else None

    def pose_point(index: int) -> Any | None:
        if (
            subset_row is None
            or getattr(candidate, "ndim", 0) != 2
            or index >= len(subset_row)
            or float(subset_row[index]) < 0
        ):
            return None
        point = np.asarray(candidate[index], dtype=np.float32)
        if point.shape != (2,) or not np.isfinite(point).all():
            return None
        return point

    def resolve_chain(
        indices: tuple[int, int, int],
        require_inside_frame: bool = False,
    ) -> tuple[list[Any], list[str]] | None:
        """Resolve a partly occluded joint chain without inventing a whole limb."""
        points = [pose_point(index) for index in indices]
        inferred: list[str] = []
        present = sum(point is not None for point in points)
        if present < 2:
            return None
        if points[0] is None:
            # A missing shoulder/hip makes the attachment point unsafe to infer.
            return None
        if points[1] is None and points[2] is not None:
            points[1] = points[0] * 0.5 + points[2] * 0.5
            inferred.append("middle")
        if points[2] is None and points[1] is not None:
            direction = points[1] - points[0]
            if float(np.linalg.norm(direction)) < 0.025:
                return None
            points[2] = np.clip(points[1] + direction * 0.82, 0.0, 1.0)
            inferred.append("endpoint")
        if any(point is None for point in points):
            return None
        resolved = [np.asarray(point, dtype=np.float32) for point in points]
        if require_inside_frame and not all(0 <= float(point[1]) < 0.96 for point in resolved):
            return None
        return resolved, inferred

    target_exposes_arms = category in {"tops", "one-pieces"} and garment_ratios.get("arms", 0.0) >= 0.014
    target_exposes_legs = category in {"bottoms", "one-pieces"} and garment_ratios.get("legs", 0.0) >= 0.025
    arm_chains = (("left", (2, 3, 4)), ("right", (5, 6, 7)))
    leg_chains = (("left", (8, 9, 10)), ("right", (11, 12, 13)))
    resolved_arms = {
        side: resolve_chain(indices)
        for side, indices in arm_chains
    } if target_exposes_arms else {}
    resolved_legs = {
        side: resolve_chain(indices, require_inside_frame=True)
        for side, indices in leg_chains
    } if target_exposes_legs else {}
    expected_arms = sum(value is not None for value in resolved_arms.values())
    expected_legs = sum(
        value is not None for value in resolved_legs.values()
    )
    detected_arms = components((labels_to_ids["arms"], labels_to_ids["hands"]))
    detected_legs = components((labels_to_ids["legs"], labels_to_ids["feet"]))

    height, width = segmentation_array.shape[:2]
    search_radius = max(3, int(min(width, height) * 0.022))
    endpoint_radius = max(5, int(min(width, height) * 0.055))
    arm_skin = np.isin(segmentation_array, (labels_to_ids["arms"], labels_to_ids["hands"]))
    hand_skin = segmentation_array == labels_to_ids["hands"]
    leg_skin = np.isin(segmentation_array, (labels_to_ids["legs"], labels_to_ids["feet"]))
    foot_skin = segmentation_array == labels_to_ids["feet"]
    source_hand_skin = None
    if source_segmentation is not None:
        source_hand_skin = np.asarray(source_segmentation) == labels_to_ids["hands"]
        if source_hand_skin.shape != hand_skin.shape:
            source_hand_skin = np.asarray(
                Image.fromarray((source_hand_skin.astype("uint8") * 255), mode="L").resize(
                    (width, height), Image.Resampling.NEAREST
                )
            ) > 0
    output_gray = None
    if output_image is not None:
        output_array = np.asarray(output_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS))
        output_gray = cv2.cvtColor(output_array, cv2.COLOR_RGB2GRAY)

    def local_count(mask: Any, point: Any, radius: int) -> int:
        x = int(float(point[0]) * width)
        y = int(float(point[1]) * height)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        return int(mask[y0:y1, x0:x1].sum())

    def segment_coverage(mask: Any, start: Any, end: Any, start_fraction: float = 0.0) -> float:
        samples = [
            start * (1.0 - fraction) + end * fraction
            for fraction in np.linspace(start_fraction, 1.0, 12)
        ]
        covered = sum(local_count(mask, point, search_radius) > 0 for point in samples)
        return round(covered / max(1, len(samples)), 3)

    source_hand_groups = pose.get("hands") if isinstance(pose, dict) else None
    output_hand_groups = output_pose.get("hands") if isinstance(output_pose, dict) else None

    def matching_hand(hand_groups: Any, wrist: Any) -> Any | None:
        if getattr(hand_groups, "ndim", 0) != 3:
            return None
        wrist_array = np.asarray(wrist, dtype=np.float32)
        closest = None
        closest_distance = float("inf")
        for hand in hand_groups:
            points = hand[(hand[:, 0] >= 0) & (hand[:, 1] >= 0)]
            if len(points) < 8:
                continue
            root = hand[0] if bool((hand[0] >= 0).all()) else points.mean(axis=0)
            distance = float(np.linalg.norm(root - wrist_array))
            if distance < closest_distance:
                closest, closest_distance = hand, distance
        return closest if closest_distance < 0.16 else None

    def keypoint_support(hand: Any | None, mask: Any | None) -> int:
        if hand is None or mask is None:
            return 0
        radius = max(3, int(min(width, height) * 0.008))
        supported = 0
        for point_value in hand:
            if not bool((point_value >= 0).all()):
                continue
            supported += local_count(mask, point_value, radius) > 0
        return supported

    def hand_structure(
        hand: Any | None,
        wrist: Any,
        elbow: Any,
    ) -> tuple[int, float, int, float, float, float, float, bool]:
        if hand is None:
            return 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, False
        valid = hand[(hand[:, 0] >= 0) & (hand[:, 1] >= 0)]
        if len(valid) < 16:
            return len(valid), 0.0, keypoint_support(hand, hand_skin), 0.0, 0.0, 0.0, 0.0, False
        forearm_length = max(0.025, float(np.linalg.norm(np.asarray(wrist) - np.asarray(elbow))))
        root = hand[0] if bool((hand[0] >= 0).all()) else valid.mean(axis=0)
        fingertip_indices = (4, 8, 12, 16, 20)
        fingertips = [hand[index] for index in fingertip_indices if bool((hand[index] >= 0).all())]
        extension = max((float(np.linalg.norm(point - root)) for point in fingertips), default=0.0)
        extension_ratio = extension / forearm_length
        hand_length = max(1e-6, float(np.linalg.norm(hand[12] - root)))
        palm_ratio = float(np.linalg.norm(hand[5] - hand[17])) / hand_length
        spread_ratio = max(
            (
                float(np.linalg.norm(first - second)) / hand_length
                for first in fingertips
                for second in fingertips
            ),
            default=0.0,
        )
        bone_lengths = []
        for chain in (
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ):
            bone_lengths.extend(
                float(np.linalg.norm(hand[end] - hand[start]))
                for start, end in zip(chain, chain[1:])
            )
        mean_bone = max(1e-6, float(np.mean(bone_lengths)))
        bone_cv = float(np.std(bone_lengths)) / mean_bone
        supported_points = keypoint_support(hand, hand_skin)
        detail_score = 0.0
        if output_gray is not None:
            xs = np.clip((valid[:, 0] * width).astype(int), 0, width - 1)
            ys = np.clip((valid[:, 1] * height).astype(int), 0, height - 1)
            margin = max(5, int(min(width, height) * 0.018))
            x0, x1 = max(0, int(xs.min()) - margin), min(width, int(xs.max()) + margin + 1)
            y0, y1 = max(0, int(ys.min()) - margin), min(height, int(ys.max()) + margin + 1)
            local_mask = hand_skin[y0:y1, x0:x1]
            if bool(local_mask.any()):
                laplacian = cv2.Laplacian(output_gray[y0:y1, x0:x1], cv2.CV_32F)
                detail_score = float(laplacian[local_mask].var())
        detail_ok = output_gray is None or 220.0 <= detail_score <= 900.0
        shape_ok = palm_ratio >= 0.34 and spread_ratio >= 0.40 and bone_cv <= 0.95
        complete = (
            0.28 <= extension_ratio <= 0.95
            and supported_points >= 18
            and detail_ok
            and shape_ok
        )
        return (
            len(valid),
            round(extension_ratio, 3),
            supported_points,
            round(detail_score, 1),
            round(palm_ratio, 3),
            round(spread_ratio, 3),
            round(bone_cv, 3),
            complete,
        )

    arm_details = []
    for side, indices in arm_chains:
        resolved = resolved_arms.get(side)
        if resolved is None:
            continue
        points, inferred = resolved
        upper_coverage = segment_coverage(arm_skin, points[0], points[1], 0.18)
        forearm_coverage = segment_coverage(arm_skin, points[1], points[2])
        coverage = round((upper_coverage + forearm_coverage) / 2.0, 3)
        wrist = points[2]
        hand_pixels = local_count(hand_skin, wrist, endpoint_radius)
        source_hand = matching_hand(source_hand_groups, wrist)
        output_hand = matching_hand(output_hand_groups, wrist)
        (
            output_hand_points,
            hand_extension_ratio,
            hand_supported_points,
            hand_detail_score,
            hand_palm_ratio,
            hand_spread_ratio,
            hand_bone_cv,
            output_hand_structured,
        ) = hand_structure(output_hand, wrist, points[1])
        source_hand_supported_points = keypoint_support(source_hand, source_hand_skin)
        source_hand_occluded = source_hand is None or (
            source_hand_skin is not None and source_hand_supported_points < 11
        )
        parser_hand_present = hand_pixels >= max(18, int(endpoint_radius * endpoint_radius * 0.040))
        # Production always supplies output_pose. The parser-only fallback keeps
        # this pure function usable in lightweight unit tests.
        hand_detected = parser_hand_present and (
            (
                output_hand_structured
                if source_hand_occluded
                else output_hand_points >= 16 and hand_supported_points >= 11
            )
            if output_pose is not None
            else True
        )
        complete = upper_coverage >= 0.58 and forearm_coverage >= 0.70 and hand_detected
        arm_details.append({
            "side": side,
            "segment_coverage": coverage,
            "upper_arm_coverage": upper_coverage,
            "forearm_coverage": forearm_coverage,
            "hand_detected": hand_detected,
            "hand_keypoints": output_hand_points,
            "hand_supported_points": hand_supported_points,
            "hand_extension_ratio": hand_extension_ratio,
            "hand_detail_score": hand_detail_score,
            "hand_palm_ratio": hand_palm_ratio,
            "hand_spread_ratio": hand_spread_ratio,
            "hand_bone_cv": hand_bone_cv,
            "source_hand_occluded": source_hand_occluded,
            "inferred_joints": inferred,
            "path": [point.tolist() for point in points],
            "complete": complete,
        })

    leg_details = []
    for side, indices in leg_chains:
        resolved = resolved_legs.get(side)
        if resolved is None:
            continue
        points, inferred = resolved
        upper_coverage = segment_coverage(leg_skin, points[0], points[1], 0.18)
        lower_coverage = segment_coverage(leg_skin, points[1], points[2])
        coverage = round((upper_coverage + lower_coverage) / 2.0, 3)
        ankle = points[2]
        foot_pixels = local_count(foot_skin, ankle, endpoint_radius)
        foot_detected = foot_pixels >= max(12, int(endpoint_radius * endpoint_radius * 0.025))
        complete = coverage >= 0.62 and foot_detected
        leg_details.append({
            "side": side,
            "segment_coverage": coverage,
            "upper_leg_coverage": upper_coverage,
            "lower_leg_coverage": lower_coverage,
            "foot_detected": foot_detected,
            "inferred_joints": inferred,
            "path": [point.tolist() for point in points],
            "complete": complete,
        })

    arms_complete = all(item["complete"] for item in arm_details) and len(arm_details) == expected_arms
    legs_complete = all(item["complete"] for item in leg_details) and len(leg_details) == expected_legs
    repair_targets: list[str] = []
    repair_paths: dict[str, list[list[float]]] = {}
    for item in arm_details:
        if item["upper_arm_coverage"] < 0.58 or item["forearm_coverage"] < 0.70:
            target = f"arm_{item['side']}"
            repair_targets.append(target)
            repair_paths[target] = item["path"]
        if not item["hand_detected"]:
            target = f"hand_{item['side']}"
            repair_targets.append(target)
            repair_paths[target] = item["path"]
    for item in leg_details:
        if item["segment_coverage"] < 0.62:
            target = f"leg_{item['side']}"
            repair_targets.append(target)
            repair_paths[target] = item["path"]
        if not item["foot_detected"]:
            target = f"foot_{item['side']}"
            repair_targets.append(target)
            repair_paths[target] = item["path"]
    return {
        "complete": arms_complete and legs_complete,
        "expected_arms": expected_arms,
        "detected_arms": detected_arms,
        "expected_legs": expected_legs,
        "detected_legs": detected_legs,
        "arms": arm_details,
        "legs": leg_details,
        "repair_targets": repair_targets,
        "repair_paths": repair_paths,
    }


def match_limb_skin_tone(
    person_image: Image.Image,
    generated_image: Image.Image,
    person_segmentation: Any,
    generated_segmentation: Any,
    labels_to_ids: dict[str, int],
    pose: Any | None = None,
) -> tuple[Image.Image, float, float, dict[str, str]]:
    """Match each generated limb segment to source skin from the same location first."""
    import cv2
    import numpy as np
    from PIL import ImageDraw, ImageFilter

    source_image = person_image.convert("RGB")
    if source_image.size != generated_image.size:
        source_image = source_image.resize(generated_image.size, Image.Resampling.LANCZOS)
    source = np.asarray(source_image)
    generated = np.asarray(generated_image.convert("RGB")).copy()
    source_segmentation = np.asarray(person_segmentation)
    target_segmentation = np.asarray(generated_segmentation)
    if source_segmentation.shape[:2] != source.shape[:2]:
        source_segmentation = np.asarray(
            Image.fromarray(source_segmentation.astype("uint8")).resize(
                (source.shape[1], source.shape[0]), Image.Resampling.NEAREST
            )
        )
    if target_segmentation.shape[:2] != generated.shape[:2]:
        target_segmentation = np.asarray(
            Image.fromarray(target_segmentation.astype("uint8")).resize(
                (generated.shape[1], generated.shape[0]), Image.Resampling.NEAREST
            )
        )

    source_skin = np.isin(
        source_segmentation,
        (labels_to_ids["arms"], labels_to_ids["hands"], labels_to_ids["legs"], labels_to_ids["feet"]),
    )
    source_face = source_segmentation == labels_to_ids["face"]
    target_mask = np.isin(
        target_segmentation,
        (labels_to_ids["arms"], labels_to_ids["hands"], labels_to_ids["legs"], labels_to_ids["feet"]),
    )
    if (not bool(source_skin.any()) and not bool(source_face.any())) or not bool(target_mask.any()):
        return generated_image.convert("RGB"), 0.0, 0.0, {}

    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(generated, cv2.COLOR_RGB2LAB).astype(np.float32)
    # Human parsing occasionally labels patterned sleeves as arms. Build a skin
    # colour anchor from the face and parser-confirmed hands, then discard source
    # limb pixels too far from either anchor before using them as a reference.
    anchor_masks = [
        source_segmentation == labels_to_ids["face"],
        source_segmentation == labels_to_ids["hands"],
    ]
    anchor_medians = [
        np.median(source_lab[mask], axis=0)
        for mask in anchor_masks
        if int(mask.sum()) >= 32
    ]
    if anchor_medians and bool(source_skin.any()):
        distances = np.stack(
            [np.linalg.norm(source_lab - anchor, axis=2) for anchor in anchor_medians],
            axis=0,
        )
        source_skin &= distances.min(axis=0) <= 34.0
    height, width = target_mask.shape
    bodies = pose.get("bodies", {}) if isinstance(pose, dict) else {}
    candidate = bodies.get("candidate")
    subset = bodies.get("subset")
    subset_row = subset[0] if getattr(subset, "ndim", 0) == 2 and len(subset) else None

    def point(index: int) -> Any | None:
        if (
            subset_row is None
            or getattr(candidate, "ndim", 0) != 2
            or index >= len(subset_row)
            or float(subset_row[index]) < 0
        ):
            return None
        value = np.asarray(candidate[index], dtype=np.float32)
        return value if value.shape == (2,) and np.isfinite(value).all() else None

    def line_region(start: Any, end: Any, line_width: int) -> Any:
        matte = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(matte)
        points = (
            (int(float(start[0]) * width), int(float(start[1]) * height)),
            (int(float(end[0]) * width), int(float(end[1]) * height)),
        )
        draw.line(points, fill=255, width=line_width)
        return np.asarray(matte) > 0

    def endpoint_region(center: Any, radius: int) -> Any:
        matte = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(matte)
        x, y = int(float(center[0]) * width), int(float(center[1]) * height)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        return np.asarray(matte) > 0

    regions: list[tuple[str, Any, Any]] = []
    limb_chains = (
        ("left_upper_arm", (2, 3), 0.105),
        ("left_forearm", (3, 4), 0.090),
        ("right_upper_arm", (5, 6), 0.105),
        ("right_forearm", (6, 7), 0.090),
        ("left_thigh", (8, 9), 0.140),
        ("left_lower_leg", (9, 10), 0.115),
        ("right_thigh", (11, 12), 0.140),
        ("right_lower_leg", (12, 13), 0.115),
    )
    side_regions: dict[str, Any] = {}
    for name, indices, width_ratio in limb_chains:
        start, end = point(indices[0]), point(indices[1])
        if start is None or end is None:
            continue
        region = line_region(start, end, max(10, int(min(width, height) * width_ratio)))
        side_name = name.split("_", 1)[0]
        side = f"{side_name}_{'arm' if 'arm' in name or 'forearm' in name else 'leg'}"
        side_regions[side] = side_regions.get(side, np.zeros_like(target_mask)) | region
        regions.append((name, region, side))
    for name, index, radius_ratio in (
        ("left_hand", 4, 0.075),
        ("right_hand", 7, 0.075),
        ("left_foot", 10, 0.090),
        ("right_foot", 13, 0.090),
    ):
        center = point(index)
        if center is None:
            continue
        region = endpoint_region(center, max(10, int(min(width, height) * radius_ratio)))
        side_name = name.split("_", 1)[0]
        side = f"{side_name}_{'arm' if 'hand' in name else 'leg'}"
        side_regions[side] = side_regions.get(side, np.zeros_like(target_mask)) | region
        regions.append((name, region, side))

    if not regions:
        regions = [("all_limbs", target_mask.copy(), "all")]
        side_regions["all"] = target_mask.copy()

    shift_sum = np.zeros_like(target_lab, dtype=np.float32)
    shift_weight = np.zeros(target_mask.shape, dtype=np.float32)
    reference_modes: dict[str, str] = {}
    before_values: list[float] = []
    minimum_reference = max(24, int(target_mask.size * 0.00012))
    global_source = source_skin if int(source_skin.sum()) >= minimum_reference else source_face

    for name, region, side in regions:
        target_region = target_mask & region
        if int(target_region.sum()) < minimum_reference:
            continue
        same_position = source_skin & target_region
        same_side = source_skin & side_regions.get(side, region)
        same_position_coverage = int(same_position.sum()) / max(1, int(target_region.sum()))
        if int(same_position.sum()) >= minimum_reference and same_position_coverage >= 0.30:
            reference = same_position
            target_reference = same_position & target_mask
            mode = "same-position"
        elif int(same_side.sum()) >= minimum_reference:
            reference = same_side
            target_reference = target_region
            mode = "same-side"
        else:
            reference = global_source
            target_reference = target_region
            mode = "global-fallback" if bool(source_skin.any()) else "face-fallback"
        if not bool(reference.any()) or not bool(target_reference.any()):
            continue
        source_median = np.median(source_lab[reference], axis=0)
        target_median = np.median(target_lab[target_reference], axis=0)
        # Correct the full measured skin cast.  The earlier ±10 chroma clamp
        # barely changed strongly red/yellow generated hands and then allowed
        # them through because the keypoints were valid.
        shift = np.clip(
            source_median - target_median,
            (-56.0, -24.0, -24.0),
            (56.0, 24.0, 24.0),
        )
        shift_sum[target_region] += shift
        shift_weight[target_region] += 1.0
        reference_modes[name] = mode
        before_values.append(float(np.linalg.norm(source_median - target_median)))

    # Blend neighbouring segment shifts before applying them, so local lighting
    # follows the arm continuously without visible bands at the elbow or wrist.
    smoothing_sigma = max(5.0, min(width, height) * 0.035)
    smooth_weight = cv2.GaussianBlur(shift_weight, (0, 0), smoothing_sigma)
    smooth_shift = np.stack(
        [cv2.GaussianBlur(shift_sum[:, :, channel], (0, 0), smoothing_sigma) for channel in range(3)],
        axis=2,
    )
    corrected_lab = target_lab.copy()
    corrected_pixels = target_mask & (smooth_weight > 0.01)
    if bool(corrected_pixels.any()):
        averaged_shift = np.zeros_like(target_lab, dtype=np.float32)
        averaged_shift[corrected_pixels] = (
            smooth_shift[corrected_pixels] / smooth_weight[corrected_pixels][:, None]
        )
        corrected_lab[corrected_pixels] = np.clip(
            corrected_lab[corrected_pixels] + averaged_shift[corrected_pixels], 0, 255
        )
    corrected = cv2.cvtColor(corrected_lab.astype("uint8"), cv2.COLOR_LAB2RGB)

    matte = Image.fromarray((corrected_pixels.astype("uint8") * 255), mode="L").filter(ImageFilter.GaussianBlur(1.0))
    output = Image.composite(Image.fromarray(corrected), Image.fromarray(generated), matte)
    before_delta = float(np.mean(before_values)) if before_values else 0.0
    after_values = []
    for name, region, side in regions:
        target_region = corrected_pixels & region
        source_region = source_skin & target_region
        if bool(target_region.any()) and bool(source_region.any()):
            after_values.append(float(np.linalg.norm(
                np.median(source_lab[source_region], axis=0)
                - np.median(corrected_lab[target_region], axis=0)
            )))
    after_delta = float(np.mean(after_values)) if after_values else before_delta
    return output, round(before_delta, 2), round(after_delta, 2), reference_modes


def verify_garment_integrity(
    segmentation: Any,
    category: str,
    labels_to_ids: dict[str, int],
) -> tuple[bool, str]:
    """Reject obvious category collapse such as a dress reverting to trousers."""
    import numpy as np

    segmentation_array = np.asarray(segmentation)
    if category == "one-pieces":
        garment_mask = segmentation_array == labels_to_ids["dress"]
        passed = float(garment_mask.mean()) >= 0.10
        return passed, "连衣裙主体连续" if passed else "连衣裙主体面积不足，可能混入原下装"
    if category == "tops":
        garment_mask = np.isin(segmentation_array, (labels_to_ids["top"], labels_to_ids["dress"]))
        passed = float(garment_mask.mean()) >= 0.035
        return passed, "上装主体完整" if passed else "上装主体面积不足"
    garment_mask = np.isin(segmentation_array, (labels_to_ids["skirt"], labels_to_ids["pants"]))
    passed = float(garment_mask.mean()) >= 0.035
    return passed, "下装主体完整" if passed else "下装主体面积不足"


def _skin_tone_is_credible(delta_before: float, delta_after: float) -> bool:
    return delta_after <= 10.0 and delta_after <= max(6.0, delta_before + 2.0)


def _match_local_repair_to_adjacent_skin(
    repaired_image: Image.Image,
    base_image: Image.Image,
    repair_skin: Any,
    base_skin: Any,
    repair_region: Any,
) -> tuple[Image.Image, float, float, bool]:
    """Match a generated hand to the nearest unchanged forearm lighting."""
    import cv2
    import numpy as np
    from PIL import ImageFilter

    target = np.asarray(repair_skin, dtype=bool) & np.asarray(repair_region, dtype=bool)
    region = np.asarray(repair_region, dtype="uint8")
    radius = max(21, int(min(repaired_image.size) * 0.085))
    if radius % 2 == 0:
        radius += 1
    ring = cv2.dilate(
        region,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)),
    ).astype(bool) & ~region.astype(bool)
    reference = np.asarray(base_skin, dtype=bool) & ring
    if int(target.sum()) < 64 or int(reference.sum()) < 64:
        return repaired_image, 0.0, 0.0, False

    repaired = np.asarray(repaired_image.convert("RGB"))
    base = np.asarray(base_image.convert("RGB").resize(repaired_image.size, Image.Resampling.LANCZOS))
    repaired_lab = cv2.cvtColor(repaired, cv2.COLOR_RGB2LAB).astype(np.float32)
    base_lab = cv2.cvtColor(base, cv2.COLOR_RGB2LAB).astype(np.float32)
    reference_median = np.median(base_lab[reference], axis=0)
    target_median = np.median(repaired_lab[target], axis=0)
    before = float(np.linalg.norm(reference_median - target_median))
    shift = np.clip(
        reference_median - target_median,
        (-64.0, -32.0, -32.0),
        (64.0, 32.0, 32.0),
    )
    shifted = repaired_lab.copy()
    shifted[target] = np.clip(shifted[target] + shift, 0, 255)
    shifted_rgb = cv2.cvtColor(shifted.astype("uint8"), cv2.COLOR_LAB2RGB)
    alpha = Image.fromarray((target.astype("uint8") * 255), mode="L").filter(
        ImageFilter.GaussianBlur(0.8)
    )
    output = Image.composite(Image.fromarray(shifted_rgb), repaired_image, alpha)
    output_lab = cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2LAB).astype(np.float32)
    after = float(np.linalg.norm(reference_median - np.median(output_lab[target], axis=0)))
    return output, round(before, 2), round(after, 2), True


def model_status(payload: dict[str, Any]) -> dict[str, Any]:
    weights_dir = Path(str(payload.get("weights_dir") or ""))
    human_parser_dir = Path(str(payload.get("human_parser_dir") or ""))
    files = []
    ready = True
    total_bytes = 0
    for relative_path, expected_size in EXPECTED_FILES.items():
        path = weights_dir / relative_path
        actual_size = path.stat().st_size if path.is_file() else None
        valid = actual_size == expected_size
        ready = ready and valid
        total_bytes += actual_size or 0
        files.append({
            "path": relative_path,
            "installed": path.is_file(),
            "size_bytes": actual_size,
            "expected_size_bytes": expected_size,
            "valid": valid,
        })
    parser_model = human_parser_dir / "model.safetensors"
    parser_config = human_parser_dir / "config.json"
    parser_valid = parser_model.is_file() and parser_model.stat().st_size == 256_146_352 and parser_config.is_file()
    ready = ready and parser_valid
    total_bytes += parser_model.stat().st_size if parser_model.is_file() else 0
    files.append({
        "path": "human-parser/model.safetensors",
        "installed": parser_model.is_file(),
        "size_bytes": parser_model.stat().st_size if parser_model.is_file() else None,
        "expected_size_bytes": 256_146_352,
        "valid": parser_valid,
    })

    cuda_available = bool(payload.get("cuda_available", False))
    gpu_name = payload.get("gpu_name") if cuda_available else None
    if payload.get("probe_cuda"):
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        except ImportError:
            cuda_available = False
            gpu_name = None
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    runtime_importable = (
        importlib.util.find_spec("fashn_vton") is not None
        and importlib.util.find_spec("fashn_human_parser") is not None
    )
    ready = ready and cuda_available and runtime_importable and torch_version is not None

    return {
        "installed": all(item["installed"] for item in files),
        "ready": ready and cuda_available and runtime_importable,
        "model_revision": MODEL_REVISION,
        "source_revision": SOURCE_REVISION,
        "human_parser_revision": HUMAN_PARSER_REVISION,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "runtime_importable": runtime_importable,
        "weights_bytes": total_bytes,
        "files": files,
        "pose_provider": "CPUExecutionProvider",
        "message": (
            "本地试衣模型已就绪"
            if ready and cuda_available and runtime_importable
            else "试衣运行时或权重不完整"
        ),
        "release_status": "evaluation-passed-license-review-required",
    }


def generate(payload: dict[str, Any]) -> dict[str, Any]:
    person_path = Path(str(payload.get("person_path") or ""))
    garment_path = Path(str(payload.get("garment_path") or ""))
    output_path = Path(str(payload.get("output_path") or ""))
    weights_dir = Path(str(payload.get("weights_dir") or ""))
    human_parser_dir = Path(str(payload.get("human_parser_dir") or ""))
    requested_category = str(payload.get("category") or "automatic")
    garment_photo_type = str(payload.get("garment_photo_type") or "model")
    sleeve_mode = str(payload.get("sleeve_mode") or "automatic")
    num_timesteps = int(payload.get("num_timesteps") or 20)
    seed = int(payload.get("seed") or 42)

    if not person_path.is_file() or not garment_path.is_file():
        raise ValueError("人物或服装素材不存在")
    if requested_category not in VALID_CATEGORIES:
        raise ValueError("服装类别无效")
    if garment_photo_type not in VALID_GARMENT_TYPES:
        raise ValueError("服装照片类型无效")
    if sleeve_mode not in VALID_SLEEVE_MODES:
        raise ValueError("袖长与遮挡模式无效")
    if num_timesteps not in {20, 30}:
        raise ValueError("生成步数只允许 20 或 30")
    if not model_status({
        "weights_dir": str(weights_dir),
        "human_parser_dir": str(human_parser_dir),
        "probe_cuda": True,
    })["ready"]:
        raise ValueError("本地试衣模型尚未就绪")

    from fashn_human_parser import FashnHumanParser

    original_parser_init = FashnHumanParser.__init__

    def pinned_parser_init(self: Any, model_id: str = "", device: str | None = None) -> None:
        original_parser_init(self, model_id=str(human_parser_dir), device=device)

    FashnHumanParser.__init__ = pinned_parser_init
    from fashn_vton import TryOnPipeline

    with Image.open(person_path) as source:
        person_image = ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(garment_path) as source:
        garment_image = ImageOps.exif_transpose(source).convert("RGB")

    started = time.perf_counter()
    pipeline = TryOnPipeline(weights_dir=str(weights_dir), device="cuda")
    analysis = inspect_tryon_inputs(
        pipeline,
        person_image,
        garment_image,
        requested_category,
        sleeve_mode,
        garment_photo_type,
    )
    category = analysis["category"]
    category_notice = analysis["category_notice"]
    adaptive_strategy = analysis["adaptive_strategy"]
    coverage_notice = analysis["coverage_notice"]
    effective_timesteps = 30 if adaptive_strategy == "reconstruct-coverage" else num_timesteps

    # Stage 1: generate the complete garment once. This remains the visual base.
    garment_result = pipeline(
        person_image=person_image,
        garment_image=garment_image,
        category=category,
        garment_photo_type=garment_photo_type,
        num_samples=1,
        num_timesteps=effective_timesteps,
        guidance_scale=1.5,
        seed=seed,
        segmentation_free=True,
    )
    if not garment_result.images:
        raise RuntimeError("服装生成阶段没有返回图片")

    from fashn_human_parser import LABELS_TO_IDS
    import numpy as np

    image = garment_result.images[0]
    generated_segmentation = pipeline.hp_model.predict(np.array(image))
    generated_pose = pipeline.pose_model(np.array(image)[..., ::-1])
    limb_quality = assess_limb_completeness(
        generated_segmentation,
        analysis["pose"],
        category,
        analysis["garment_ratios"],
        LABELS_TO_IDS,
        generated_pose,
        image,
        analysis["person_segmentation"],
    )
    limbs_merged = False
    generation_passes = 1
    repair_attempted = False

    # Stage 2: run the expensive reconstruction pass only when the garment-first
    # image is actually missing an expected arm/leg, or when the user forces it.
    if adaptive_strategy == "reconstruct-coverage" and (
        not limb_quality["complete"] or sleeve_mode == "reveal-arms"
    ):
        repair_attempted = True
        reconstruction_result = pipeline(
            person_image=person_image,
            garment_image=garment_image,
            category=category,
            garment_photo_type=garment_photo_type,
            num_samples=1,
            num_timesteps=effective_timesteps,
            guidance_scale=1.5,
            # Use a deterministic alternate sample for repair. Repeating the
            # garment pass seed tends to reproduce the same malformed hand.
            seed=seed + 104729,
            segmentation_free=False,
        )
        if not reconstruction_result.images:
            raise RuntimeError("肢体补全阶段没有返回图片")
        reconstruction_image = reconstruction_result.images[0]
        reconstruction_segmentation = pipeline.hp_model.predict(np.array(reconstruction_image))
        base_arms = {item["side"]: item for item in limb_quality["arms"]}
        base_image = image
        base_segmentation = generated_segmentation
        base_pose = generated_pose
        base_quality = limb_quality
        preview, preview_merged = merge_targeted_limb_repairs(
            image,
            reconstruction_image,
            reconstruction_segmentation,
            limb_quality["repair_targets"],
            analysis["pose"],
            LABELS_TO_IDS,
            limb_quality["repair_paths"],
        )
        generation_passes = 2
        if preview_merged:
            preview_segmentation = pipeline.hp_model.predict(np.array(preview))
            preview_pose = pipeline.pose_model(np.array(preview)[..., ::-1])
            preview_quality = assess_limb_completeness(
                preview_segmentation,
                analysis["pose"],
                category,
                analysis["garment_ratios"],
                LABELS_TO_IDS,
                preview_pose,
                preview,
                analysis["person_segmentation"],
            )
            preview_arms = {item["side"]: item for item in preview_quality["arms"]}
            accepted_targets = []
            for target in base_quality["repair_targets"]:
                kind, side = target.split("_", 1)
                base_arm = base_arms.get(side, {})
                preview_arm = preview_arms.get(side, {})
                if kind == "hand":
                    improved = (
                        preview_arm.get("hand_detected", False)
                        or preview_arm.get("hand_detail_score", 0.0)
                        >= base_arm.get("hand_detail_score", 0.0) + 35.0
                    )
                elif kind == "arm":
                    improved = preview_arm.get("segment_coverage", 0.0) > base_arm.get("segment_coverage", 0.0)
                else:
                    improved = True
                if improved:
                    accepted_targets.append(target)

            if accepted_targets:
                image, limbs_merged = merge_targeted_limb_repairs(
                    base_image,
                    reconstruction_image,
                    reconstruction_segmentation,
                    accepted_targets,
                    analysis["pose"],
                    LABELS_TO_IDS,
                    {target: base_quality["repair_paths"][target] for target in accepted_targets},
                )
                generated_segmentation = pipeline.hp_model.predict(np.array(image))
                generated_pose = pipeline.pose_model(np.array(image)[..., ::-1])
                limb_quality = assess_limb_completeness(
                    generated_segmentation,
                    analysis["pose"],
                    category,
                    analysis["garment_ratios"],
                    LABELS_TO_IDS,
                    generated_pose,
                    image,
                    analysis["person_segmentation"],
                )
            else:
                image = base_image
                generated_segmentation = base_segmentation
                generated_pose = base_pose
                limb_quality = base_quality

    # Stage 3: use the person's visible source skin as the colour reference for
    # generated limbs. This corrects dark or differently tinted reconstructed arms
    # while retaining their local shading.
    image, skin_delta_before, skin_delta_after, skin_reference_modes = match_limb_skin_tone(
        person_image,
        image,
        analysis["person_segmentation"],
        generated_segmentation,
        LABELS_TO_IDS,
        analysis["pose"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".partial.png")

    # The upstream VTON preprocessor preserves hands only in the conditioning
    # input. Restore locally recognised hands and handheld items after sampling
    # so the diffusion pass cannot redraw fingers or a phone in front of clothing.
    image, hands_preserved = preserve_hand_foreground(
        person_image,
        image,
        analysis["person_segmentation"],
        LABELS_TO_IDS["hands"],
        LABELS_TO_IDS["bag"],
        analysis["pose"],
    )

    # Stage 4: parse the actual composited output again. Verification must assess
    # the saved image, including restored source hands and colour correction.
    final_segmentation = pipeline.hp_model.predict(np.array(image))
    final_pose = pipeline.pose_model(np.array(image)[..., ::-1])
    limb_quality = assess_limb_completeness(
        final_segmentation,
        analysis["pose"],
        category,
        analysis["garment_ratios"],
        LABELS_TO_IDS,
        final_pose,
        image,
        analysis["person_segmentation"],
    )
    garment_ok, garment_check = verify_garment_integrity(
        final_segmentation,
        category,
        LABELS_TO_IDS,
    )
    verification_passed = (
        garment_ok
        and limb_quality["complete"]
        and _skin_tone_is_credible(skin_delta_before, skin_delta_after)
    )
    image.save(temporary_path, format="PNG", optimize=True)
    temporary_path.replace(output_path)
    width, height = image.size
    return {
        "output_path": str(output_path.resolve()),
        "width": width,
        "height": height,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "category": category,
        "requested_category": requested_category,
        "category_notice": category_notice,
        "garment_photo_type": garment_photo_type,
        "sleeve_mode": sleeve_mode,
        "segmentation_free": True,
        "adaptive_strategy": adaptive_strategy,
        "coverage_notice": (
            f"{coverage_notice}"
            f"{' 第一阶段已检测到所需四肢，已跳过重复生成。' if adaptive_strategy == 'reconstruct-coverage' and not repair_attempted else ''}"
            f"{' 已运行局部肢体修补并只合并缺失区域。' if limbs_merged else ''}"
            f"{' 修补候选未通过合并条件，已保留服装生成结果。' if repair_attempted and not limbs_merged else ''}"
            f" 已按原图可见皮肤校正生成肢体的肤色。"
            f"{' 已锁定原图中经像素验证确实可见的手部与手机。' if hands_preserved else ''}"
        ),
        "num_timesteps": effective_timesteps,
        "generation_passes": generation_passes,
        "skin_tone_delta_before": skin_delta_before,
        "skin_tone_delta_after": skin_delta_after,
        "verification_status": "passed" if verification_passed else "review",
        "quality_checks": {
            "garment": {"passed": garment_ok, "message": garment_check},
            "limbs": limb_quality,
            "skin_tone": {
                "passed": _skin_tone_is_credible(skin_delta_before, skin_delta_after),
                "delta_before": skin_delta_before,
                "delta_after": skin_delta_after,
                "reference_modes": skin_reference_modes,
            },
            "phone_preserved": hands_preserved,
        },
        "seed": seed,
        "device": "cuda",
        "model_revision": MODEL_REVISION,
        "source_revision": SOURCE_REVISION,
        "human_parser_revision": HUMAN_PARSER_REVISION,
    }


def _hand_repair_candidate_is_credible(
    target: dict[str, Any],
    skin_delta_before: float,
    skin_delta_after: float,
) -> bool:
    return (
        int(target["hand_supported_points"]) >= 18
        and 0.35 <= float(target["hand_extension_ratio"]) <= 0.95
        and 220.0 <= float(target["hand_detail_score"]) <= 900.0
        and float(target.get("hand_palm_ratio", 0.0)) >= 0.34
        and float(target.get("hand_spread_ratio", 0.0)) >= 0.40
        and float(target.get("hand_bone_cv", 99.0)) <= 0.95
        and _skin_tone_is_credible(skin_delta_before, skin_delta_after)
    )


def _hand_repair_candidate_score(target: dict[str, Any], skin_delta: float) -> float:
    return (
        float(target["hand_supported_points"]) * 40.0
        - abs(float(target["hand_detail_score"]) - 620.0) * 0.55
        - abs(float(target["hand_extension_ratio"]) - 0.75) * 120.0
        - skin_delta * 3.0
    )


def finalize_hand_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    """Select and merge only hand candidates that pass the existing quality gate."""
    from fashn_human_parser import FashnHumanParser, LABELS_TO_IDS
    import cv2
    import numpy as np
    from PIL import ImageFilter

    person_path = Path(str(payload.get("person_path") or ""))
    garment_path = Path(str(payload.get("garment_path") or ""))
    output_path = Path(str(payload.get("output_path") or ""))
    weights_dir = Path(str(payload.get("weights_dir") or ""))
    human_parser_dir = Path(str(payload.get("human_parser_dir") or ""))
    repair_groups = payload.get("repair_groups")
    if not person_path.is_file() or not garment_path.is_file() or not output_path.is_file():
        raise ValueError("手部候选核验缺少原图、服装图或基础结果")
    if not isinstance(repair_groups, list):
        raise ValueError("手部候选列表无效")

    original_parser_init = FashnHumanParser.__init__

    def pinned_parser_init(self: Any, model_id: str = "", device: str | None = None) -> None:
        original_parser_init(self, model_id=str(human_parser_dir), device=device)

    FashnHumanParser.__init__ = pinned_parser_init
    from fashn_vton import TryOnPipeline

    person = Image.open(person_path).convert("RGB")
    garment = Image.open(garment_path).convert("RGB")
    current = Image.open(output_path).convert("RGB")
    started = time.perf_counter()
    pipeline = TryOnPipeline(weights_dir=str(weights_dir), device="cuda")
    analysis = inspect_tryon_inputs(
        pipeline,
        person,
        garment,
        str(payload.get("category") or "automatic"),
        str(payload.get("sleeve_mode") or "automatic"),
        str(payload.get("garment_photo_type") or "model"),
    )
    accepted_sides: list[str] = []
    candidate_evaluations: list[dict[str, Any]] = []
    last_skin = (0.0, 0.0, {})
    phone_preserved = False

    for group in repair_groups:
        if not isinstance(group, dict):
            continue
        side = str(group.get("side") or "")
        mask_path = Path(str(group.get("mask_path") or ""))
        candidate_paths = [Path(str(value)) for value in group.get("candidate_paths", [])]
        if side not in {"left", "right"} or not mask_path.is_file():
            continue
        region = np.asarray(Image.open(mask_path).convert("L").resize(current.size, Image.Resampling.NEAREST)) > 0
        base_array = np.asarray(current)
        base_segmentation = np.asarray(pipeline.hp_model.predict(base_array))
        base_skin = np.isin(base_segmentation, (LABELS_TO_IDS["arms"], LABELS_TO_IDS["hands"]))
        base_lab = cv2.cvtColor(base_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        anchor_size = max(31, int(min(current.size) * 0.11))
        if anchor_size % 2 == 0:
            anchor_size += 1
        adjacent_ring = cv2.dilate(
            region.astype("uint8"),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (anchor_size, anchor_size)),
        ).astype(bool) & ~region
        adjacent_skin = base_skin & adjacent_ring
        best: tuple[float, Image.Image, dict[str, Any], tuple[float, float, dict[str, str]], bool] | None = None

        for candidate_path in candidate_paths:
            if not candidate_path.is_file():
                continue
            candidate = Image.open(candidate_path).convert("RGB").resize(current.size, Image.Resampling.LANCZOS)
            candidate_segmentation = np.asarray(pipeline.hp_model.predict(np.asarray(candidate)))
            candidate_skin = np.isin(
                candidate_segmentation,
                (LABELS_TO_IDS["arms"], LABELS_TO_IDS["hands"]),
            )
            # Keep only the generated limb's main connected skin component.  The
            # inpainting model can leave thin coloured guide lines inside the
            # broad repair region; expanding the parser mask used to pull those
            # lines into the final image around the wrist and fingers.
            raw_candidate_skin = candidate_skin & region
            candidate_lab = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2LAB)
            skin_pixels = candidate_lab[raw_candidate_skin]
            if int(adjacent_skin.sum()) >= 64:
                median_chroma = np.median(base_lab[adjacent_skin][:, 1:3], axis=0)
                chroma_limit = 24.0
            elif len(skin_pixels) >= 64:
                median_chroma = np.median(skin_pixels[:, 1:3], axis=0)
                chroma_limit = 18.0
            else:
                median_chroma = None
                chroma_limit = 0.0
            if median_chroma is not None:
                chroma_distance = np.linalg.norm(
                    candidate_lab[:, :, 1:3].astype(np.float32) - median_chroma,
                    axis=2,
                )
                raw_candidate_skin &= chroma_distance <= chroma_limit
            candidate_gray = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2GRAY)
            local_edges = (
                np.abs(cv2.Laplacian(candidate_gray, cv2.CV_32F)) > 18.0
            ).astype(np.float32)
            texture_window = max(9, int(round(min(current.size) / 44)))
            if texture_window % 2 == 0:
                texture_window += 1
            edge_density = cv2.boxFilter(
                local_edges,
                -1,
                (texture_window, texture_window),
                normalize=True,
            )
            raw_candidate_skin &= edge_density <= 0.30
            clean_size = max(5, min(9, int(round(min(current.size) / 90))))
            if clean_size % 2 == 0:
                clean_size += 1
            clean_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (clean_size, clean_size),
            )
            raw_candidate_skin = cv2.morphologyEx(
                raw_candidate_skin.astype("uint8"),
                cv2.MORPH_OPEN,
                clean_kernel,
            )
            component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
                raw_candidate_skin,
                8,
            )
            alpha_mask = np.zeros_like(region)
            if component_count > 1:
                largest_component = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
                alpha_mask = component_labels == largest_component
            contours, _ = cv2.findContours(
                alpha_mask.astype("uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            filled_mask = np.zeros_like(alpha_mask, dtype="uint8")
            cv2.drawContours(filled_mask, contours, -1, 1, cv2.FILLED)
            seam_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            alpha_mask = cv2.morphologyEx(
                filled_mask,
                cv2.MORPH_CLOSE,
                seam_kernel,
            ).astype(bool)
            alpha = Image.fromarray((alpha_mask.astype("uint8") * 255), mode="L")
            alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.0))
            merged = Image.composite(candidate, current, alpha)
            merged_segmentation = pipeline.hp_model.predict(np.asarray(merged))
            merged, before_delta, after_delta, reference_modes = match_limb_skin_tone(
                person,
                merged,
                analysis["person_segmentation"],
                merged_segmentation,
                LABELS_TO_IDS,
                analysis["pose"],
            )
            merged, local_before, local_after, local_matched = _match_local_repair_to_adjacent_skin(
                merged,
                current,
                alpha_mask,
                base_skin,
                region,
            )
            if local_matched:
                before_delta, after_delta = local_before, local_after
                reference_modes[f"{side}_hand"] = "adjacent-forearm"
            merged, candidate_phone_preserved = preserve_hand_foreground(
                person,
                merged,
                analysis["person_segmentation"],
                LABELS_TO_IDS["hands"],
                LABELS_TO_IDS["bag"],
                analysis["pose"],
            )
            # Restore the clean base dress/background wherever the inpainting
            # candidate introduced pale-blue guide texture around the new hand.
            merged_array = np.asarray(merged).copy()
            merged_lab = cv2.cvtColor(merged_array, cv2.COLOR_RGB2LAB)
            guide_texture = region & (merged_lab[:, :, 1] < 138) & (merged_lab[:, :, 2] < 138)
            merged_array[guide_texture] = base_array[guide_texture]
            merged = Image.fromarray(merged_array)

            final_segmentation = pipeline.hp_model.predict(np.asarray(merged))
            final_pose = pipeline.pose_model(np.asarray(merged)[..., ::-1])
            limbs = assess_limb_completeness(
                final_segmentation,
                analysis["pose"],
                analysis["category"],
                analysis["garment_ratios"],
                LABELS_TO_IDS,
                final_pose,
                merged,
                analysis["person_segmentation"],
            )
            garment_ok, _ = verify_garment_integrity(final_segmentation, analysis["category"], LABELS_TO_IDS)
            target = next((arm for arm in limbs["arms"] if arm["side"] == side), None)
            repaired_hand_is_credible = target is not None and _hand_repair_candidate_is_credible(
                target,
                before_delta,
                after_delta,
            )
            candidate_evaluations.append({
                "side": side,
                "candidate": candidate_path.stem.rsplit("-", 1)[-1],
                "garment_passed": garment_ok,
                "skin_delta_before": before_delta,
                "skin_delta_after": after_delta,
                "hand": target,
                "accepted": bool(target is not None and target["complete"] and repaired_hand_is_credible),
            })
            if target is None or not garment_ok or not target["complete"] or not repaired_hand_is_credible:
                continue
            score = _hand_repair_candidate_score(target, after_delta)
            if best is None or score > best[0]:
                best = (
                    score,
                    merged,
                    limbs,
                    (before_delta, after_delta, reference_modes),
                    candidate_phone_preserved,
                )
        if best is not None:
            current = best[1]
            accepted_sides.append(side)
            last_skin = best[3]
            phone_preserved = phone_preserved or best[4]

    final_segmentation = pipeline.hp_model.predict(np.asarray(current))
    final_pose = pipeline.pose_model(np.asarray(current)[..., ::-1])
    limb_quality = assess_limb_completeness(
        final_segmentation,
        analysis["pose"],
        analysis["category"],
        analysis["garment_ratios"],
        LABELS_TO_IDS,
        final_pose,
        current,
        analysis["person_segmentation"],
    )
    garment_ok, garment_message = verify_garment_integrity(
        final_segmentation,
        analysis["category"],
        LABELS_TO_IDS,
    )
    before_delta, after_delta, reference_modes = last_skin
    if accepted_sides:
        temporary_path = output_path.with_suffix(".hand-repair.partial.png")
        current.save(temporary_path, optimize=True)
        temporary_path.replace(output_path)
    verification_passed = (
        garment_ok
        and limb_quality["complete"]
        and bool(accepted_sides)
        and _skin_tone_is_credible(before_delta, after_delta)
    )
    return {
        "repair_applied": bool(accepted_sides),
        "accepted_sides": accepted_sides,
        "candidate_evaluations": candidate_evaluations,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "verification_status": "passed" if verification_passed else "review",
        "quality_checks": {
            "garment": {"passed": garment_ok, "message": garment_message},
            "limbs": limb_quality,
            "skin_tone": {
                "passed": not accepted_sides or _skin_tone_is_credible(before_delta, after_delta),
                "delta_before": before_delta,
                "delta_after": after_delta,
                "reference_modes": reference_modes,
            },
            "phone_preserved": phone_preserved,
        },
    }


def respond(request: dict[str, Any]) -> dict[str, Any]:
    request_id = str(request.get("request_id") or uuid.uuid4())
    if request.get("protocol_version") != PROTOCOL_VERSION:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "success": False,
            "error": {"code": "protocol_mismatch", "message": "Unsupported protocol version"},
        }

    payload = request.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    try:
        if request.get("op") == "tryon.status":
            result = model_status(payload)
        elif request.get("op") == "tryon.analyze":
            result = analyze_tryon_scene(payload)
        elif request.get("op") == "tryon.generate":
            result = generate(payload)
        elif request.get("op") == "tryon.finalize_hand_candidates":
            result = finalize_hand_candidates(payload)
        else:
            raise ValueError(f"未知操作：{request.get('op')}")
    except Exception as exc:
        torch_module = sys.modules.get("torch")
        out_of_memory = getattr(torch_module, "OutOfMemoryError", None) if torch_module else None
        if out_of_memory is not None and isinstance(exc, out_of_memory):
            return failure(request_id, "gpu_out_of_memory", f"显存不足：{exc}")
        return failure(request_id, "tryon_failed", str(exc))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "success": True,
        "payload": result,
    }


def failure(request_id: str, code: str, message: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "success": False,
        "error": {"code": code, "message": message},
    }


def run_once() -> int:
    line = sys.stdin.readline()
    if not line:
        return 2
    try:
        request = json.loads(line)
        response = respond(request)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        response = failure("unknown", "invalid_request", str(exc))
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_once())
