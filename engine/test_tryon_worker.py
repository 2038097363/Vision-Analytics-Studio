import tempfile
import unittest
from pathlib import Path

from PIL import Image

from engine.tryon_worker import (
    _hand_repair_candidate_is_credible,
    _hand_repair_candidate_score,
    _match_local_repair_to_adjacent_skin,
    _skin_tone_is_credible,
    assess_limb_completeness,
    choose_automatic_strategy,
    choose_garment_category,
    evaluate_tryon_scene,
    match_limb_skin_tone,
    merge_targeted_limb_repairs,
    model_status,
    preserve_hand_foreground,
    respond,
)


class TryOnWorkerTests(unittest.TestCase):
    def test_hand_repair_rejects_high_frequency_guide_texture(self) -> None:
        noisy = {
            "hand_supported_points": 21,
            "hand_extension_ratio": 0.78,
            "hand_detail_score": 1066.0,
            "hand_palm_ratio": 0.42,
            "hand_spread_ratio": 0.62,
            "hand_bone_cv": 0.55,
        }
        self.assertFalse(_hand_repair_candidate_is_credible(noisy, 5.0, 5.0))

    def test_hand_repair_rejects_narrow_fused_fingers(self) -> None:
        fused = {
            "hand_supported_points": 20,
            "hand_extension_ratio": 0.79,
            "hand_detail_score": 674.0,
            "hand_palm_ratio": 0.316,
            "hand_spread_ratio": 0.428,
            "hand_bone_cv": 0.634,
        }
        self.assertFalse(_hand_repair_candidate_is_credible(fused, 5.0, 5.0))

    def test_hand_repair_prefers_natural_detail_over_noisy_detail(self) -> None:
        natural = {
            "hand_supported_points": 20,
            "hand_extension_ratio": 0.77,
            "hand_detail_score": 530.0,
        }
        noisy = {
            "hand_supported_points": 20,
            "hand_extension_ratio": 0.77,
            "hand_detail_score": 880.0,
        }
        self.assertGreater(
            _hand_repair_candidate_score(natural, 15.0),
            _hand_repair_candidate_score(noisy, 15.0),
        )

    def test_missing_weights_are_reported_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = model_status({
                "weights_dir": str(Path(directory) / "weights"),
                "human_parser_dir": str(Path(directory) / "parser"),
            })
        self.assertFalse(result["installed"])
        self.assertFalse(result["ready"])
        self.assertTrue(all(not item["installed"] for item in result["files"]))

    def test_layered_model_outfit_is_blocked_before_generation(self) -> None:
        result = evaluate_tryon_scene(
            {"face": 0.02, "hair": 0.03, "dress": 0.18, "arms": 0.04, "hands": 0.02, "legs": 0.04},
            {"top": 0.228, "dress": 0.068},
        )
        self.assertFalse(result["supported"])
        self.assertTrue(result["layered_reference"])

    def test_distant_person_is_blocked_before_generation(self) -> None:
        result = evaluate_tryon_scene(
            {"face": 0.007, "hair": 0.024, "top": 0.033, "dress": 0.026, "hands": 0.001},
            {"dress": 0.10},
        )
        self.assertFalse(result["supported"])
        self.assertLess(result["person_coverage"], 0.12)

    def test_close_person_with_single_garment_is_supported(self) -> None:
        result = evaluate_tryon_scene(
            {"face": 0.025, "hair": 0.05, "top": 0.16, "arms": 0.07, "hands": 0.025, "pants": 0.08},
            {"dress": 0.19, "arms": 0.025, "legs": 0.04},
        )
        self.assertTrue(result["supported"])
        self.assertFalse(result["layered_reference"])
        self.assertEqual(result["blockers"], [])

    def test_skin_tone_gate_rejects_a_visibly_worse_correction(self) -> None:
        self.assertFalse(_skin_tone_is_credible(4.97, 14.69))
        self.assertTrue(_skin_tone_is_credible(15.0, 6.0))

    def test_local_hand_tone_uses_adjacent_forearm(self) -> None:
        import numpy as np

        base = Image.new("RGB", (80, 80), (210, 170, 150))
        repaired = Image.new("RGB", (80, 80), (210, 170, 150))
        for y in range(38, 66):
            for x in range(34, 50):
                repaired.putpixel((x, y), (125, 78, 62))
        region = np.zeros((80, 80), dtype=bool)
        region[34:70, 28:56] = True
        repair_skin = np.zeros_like(region)
        repair_skin[38:66, 34:50] = True
        base_skin = np.zeros_like(region)
        base_skin[18:34, 34:50] = True
        corrected, before, after, matched = _match_local_repair_to_adjacent_skin(
            repaired,
            base,
            repair_skin,
            base_skin,
            region,
        )
        self.assertTrue(matched)
        self.assertLess(after, before)
        self.assertGreater(sum(corrected.getpixel((40, 50))), sum(repaired.getpixel((40, 50))))

    def test_invalid_category_fails_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            person = root / "person.png"
            garment = root / "garment.png"
            Image.new("RGB", (384, 512), "white").save(person)
            Image.new("RGB", (384, 512), "black").save(garment)
            response = respond({
                "protocol_version": 1,
                "request_id": "invalid-category",
                "op": "tryon.generate",
                "payload": {
                    "person_path": str(person),
                    "garment_path": str(garment),
                    "output_path": str(root / "result.png"),
                    "weights_dir": str(root / "weights"),
                    "human_parser_dir": str(root / "parser"),
                    "category": "shoes",
                },
            })
        self.assertFalse(response["success"])
        self.assertEqual(response["error"]["code"], "tryon_failed")
        self.assertIn("类别", response["error"]["message"])

    def test_automatic_strategy_reconstructs_covered_arms_for_tops(self) -> None:
        strategy, _ = choose_automatic_strategy(
            "tops",
            {"top": 0.16, "arms": 0.004, "hands": 0.001, "bag": 0.0},
            {"top": 0.14, "arms": 0.04},
            0.42,
        )
        self.assertEqual(strategy, "reconstruct-coverage")

    def test_automatic_strategy_reconstructs_one_piece_with_handheld_item(self) -> None:
        strategy, reason = choose_automatic_strategy(
            "one-pieces",
            {"top": 0.13, "pants": 0.12, "hands": 0.015, "bag": 0.01},
            {"dress": 0.22, "arms": 0.04},
            0.42,
        )
        self.assertEqual(strategy, "reconstruct-coverage")
        self.assertIn("手持手机", reason)

    def test_automatic_strategy_reconstructs_covered_legs_for_bottoms(self) -> None:
        strategy, _ = choose_automatic_strategy(
            "bottoms",
            {"pants": 0.18, "legs": 0.004},
            {"skirt": 0.10, "legs": 0.07},
            0.42,
        )
        self.assertEqual(strategy, "reconstruct-coverage")

    def test_garment_category_detects_one_piece_before_model_loading(self) -> None:
        category, _ = choose_garment_category({"dress": 0.18, "top": 0.06, "pants": 0.0})
        self.assertEqual(category, "one-pieces")

    def test_hand_foreground_is_restored_after_generation(self) -> None:
        import numpy as np

        source = Image.new("RGB", (12, 12), "white")
        source.putpixel((6, 6), (210, 140, 100))
        generated = Image.new("RGB", (12, 12), "navy")
        segmentation = np.zeros((12, 12), dtype=np.uint8)
        segmentation[6, 6] = 12
        segmentation[6, 7] = 8

        restored, preserved = preserve_hand_foreground(source, generated, segmentation, 12, 8)

        self.assertTrue(preserved)
        self.assertNotEqual(restored.getpixel((6, 6)), generated.getpixel((6, 6)))

    def test_upper_hand_pose_restores_a_phone_without_bag_label(self) -> None:
        import numpy as np

        source = Image.new("RGB", (100, 120), "white")
        source.putpixel((77, 25), (230, 80, 70))
        generated = Image.new("RGB", (100, 120), "navy")
        segmentation = np.zeros((120, 100), dtype=np.uint8)
        segmentation[20:42, 68:86] = 12
        upper_hand = np.array([[[0.78, 0.25]] * 21, [[0.45, 0.82]] * 21])

        restored, preserved = preserve_hand_foreground(
            source,
            generated,
            segmentation,
            12,
            8,
            {"hands": upper_hand},
        )

        self.assertTrue(preserved)
        self.assertNotEqual(restored.getpixel((77, 25)), generated.getpixel((77, 25)))

    def test_one_piece_reconstructs_exposed_arm_when_lower_parser_is_sparse(self) -> None:
        strategy, _ = choose_automatic_strategy(
            "one-pieces",
            {"top": 0.16, "pants": 0.02, "arms": 0.003},
            {"dress": 0.20, "arms": 0.05},
            0.42,
        )
        self.assertEqual(strategy, "reconstruct-coverage")

    def test_targeted_limb_repair_keeps_garment_pixels_outside_missing_arm(self) -> None:
        import numpy as np

        garment_base = Image.new("RGB", (100, 100), "white")
        reconstruction = Image.new("RGB", (100, 100), "red")
        segmentation = np.zeros((100, 100), dtype=np.uint8)
        segmentation[15:85, 15:30] = 12
        candidate = np.zeros((18, 2), dtype=np.float32)
        candidate[2], candidate[3], candidate[4] = (0.22, 0.2), (0.22, 0.5), (0.22, 0.8)
        merged, did_merge = merge_targeted_limb_repairs(
            garment_base,
            reconstruction,
            segmentation,
            ["arm_left"],
            {"bodies": {"candidate": candidate}},
            {"arms": 12, "hands": 13, "legs": 14, "feet": 15},
        )

        self.assertTrue(did_merge)
        self.assertEqual(merged.getpixel((0, 0)), garment_base.getpixel((0, 0)))
        self.assertNotEqual(merged.getpixel((22, 50)), garment_base.getpixel((22, 50)))

    def test_skin_tone_matching_uses_visible_source_skin(self) -> None:
        import numpy as np

        source = Image.new("RGB", (20, 20), (232, 188, 166))
        generated = Image.new("RGB", (20, 20), (126, 82, 64))
        source_segmentation = np.full((20, 20), 12, dtype=np.uint8)
        generated_segmentation = np.full((20, 20), 12, dtype=np.uint8)
        corrected, before, after, reference_modes = match_limb_skin_tone(
            source,
            generated,
            source_segmentation,
            generated_segmentation,
            {"face": 1, "arms": 12, "hands": 13, "legs": 14, "feet": 15},
        )

        self.assertGreater(sum(corrected.getpixel((10, 10))), sum(generated.getpixel((10, 10))))
        self.assertLess(after, before)
        self.assertEqual(reference_modes["all_limbs"], "same-position")

    def test_lower_visible_source_hand_is_preserved_without_copying_clothing(self) -> None:
        import numpy as np

        source = Image.new("RGB", (100, 120), "white")
        generated = Image.new("RGB", (100, 120), "navy")
        segmentation = np.zeros((120, 100), dtype=np.uint8)
        segmentation[92:108, 42:58] = 12
        segmentation[82:92, 42:58] = 4
        hand = np.array([[[0.50 + (index % 5) * 0.006, 0.82 + (index // 5) * 0.014] for index in range(21)]])

        restored, preserved = preserve_hand_foreground(
            source,
            generated,
            segmentation,
            12,
            8,
            {"hands": hand},
        )

        self.assertTrue(preserved)
        self.assertNotEqual(restored.getpixel((50, 100)), generated.getpixel((50, 100)))
        self.assertEqual(restored.getpixel((50, 86)), generated.getpixel((50, 86)))

    def test_limb_completeness_accepts_two_detected_arms(self) -> None:
        import numpy as np

        segmentation = np.zeros((100, 100), dtype=np.uint8)
        segmentation[20:65, 15:25] = 12
        segmentation[20:65, 75:85] = 12
        segmentation[58:70, 12:28] = 13
        segmentation[58:70, 72:88] = 13
        candidate = np.zeros((18, 2), dtype=np.float32)
        candidate[2], candidate[3], candidate[4] = (0.20, 0.20), (0.20, 0.40), (0.20, 0.62)
        candidate[5], candidate[6], candidate[7] = (0.80, 0.20), (0.80, 0.40), (0.80, 0.62)
        subset = np.arange(18, dtype=np.float32)[None, :]
        quality = assess_limb_completeness(
            segmentation,
            {"bodies": {"candidate": candidate, "subset": subset}},
            "tops",
            {"arms": 0.05},
            {"arms": 12, "hands": 13, "legs": 14, "feet": 15},
        )

        self.assertTrue(quality["complete"])
        self.assertEqual(quality["detected_arms"], 2)

    def test_limb_completeness_flags_missing_hand_separately(self) -> None:
        import numpy as np

        segmentation = np.zeros((100, 100), dtype=np.uint8)
        segmentation[20:70, 15:25] = 12
        candidate = np.zeros((18, 2), dtype=np.float32)
        candidate[2], candidate[3], candidate[4] = (0.20, 0.20), (0.20, 0.42), (0.20, 0.66)
        subset = np.full((1, 18), -1, dtype=np.float32)
        subset[0, 2:5] = (2, 3, 4)
        quality = assess_limb_completeness(
            segmentation,
            {"bodies": {"candidate": candidate, "subset": subset}},
            "tops",
            {"arms": 0.05},
            {"arms": 12, "hands": 13, "legs": 14, "feet": 15},
        )

        self.assertFalse(quality["complete"])
        self.assertIn("hand_left", quality["repair_targets"])

    def test_limb_completeness_rejects_parser_blob_without_hand_structure(self) -> None:
        import numpy as np

        segmentation = np.zeros((100, 100), dtype=np.uint8)
        segmentation[20:68, 15:25] = 12
        segmentation[62:72, 15:25] = 13
        candidate = np.zeros((18, 2), dtype=np.float32)
        candidate[2], candidate[3], candidate[4] = (0.20, 0.20), (0.20, 0.42), (0.20, 0.66)
        subset = np.full((1, 18), -1, dtype=np.float32)
        subset[0, 2:5] = (2, 3, 4)
        quality = assess_limb_completeness(
            segmentation,
            {"bodies": {"candidate": candidate, "subset": subset}},
            "tops",
            {"arms": 0.05},
            {"arms": 12, "hands": 13, "legs": 14, "feet": 15},
            {"hands": np.empty((0, 21, 2), dtype=np.float32)},
        )

        self.assertFalse(quality["arms"][0]["hand_detected"])
        self.assertIn("hand_left", quality["repair_targets"])

    def test_limb_completeness_infers_occluded_wrist_for_validation(self) -> None:
        import numpy as np

        segmentation = np.zeros((100, 100), dtype=np.uint8)
        segmentation[20:36, 15:25] = 12
        candidate = np.zeros((18, 2), dtype=np.float32)
        candidate[2], candidate[3] = (0.20, 0.20), (0.20, 0.43)
        subset = np.full((1, 18), -1, dtype=np.float32)
        subset[0, 2:4] = (2, 3)

        quality = assess_limb_completeness(
            segmentation,
            {"bodies": {"candidate": candidate, "subset": subset}},
            "tops",
            {"arms": 0.05},
            {"arms": 12, "hands": 13, "legs": 14, "feet": 15},
        )

        self.assertEqual(quality["expected_arms"], 1)
        self.assertEqual(quality["arms"][0]["inferred_joints"], ["endpoint"])
        self.assertIn("arm_left", quality["repair_targets"])
        self.assertIn("hand_left", quality["repair_targets"])
        self.assertEqual(len(quality["repair_paths"]["hand_left"]), 3)



if __name__ == "__main__":
    unittest.main()
