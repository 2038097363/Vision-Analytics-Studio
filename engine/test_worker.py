import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from engine.worker import image_preflight, person_presence_check, respond, safe_category_for_imagenet_index


class ImagePreflightTests(unittest.TestCase):
    def test_valid_local_image_returns_metrics_and_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "person.png"
            image = Image.new("RGB", (900, 1200), "white")
            draw = ImageDraw.Draw(image)
            for offset in range(80, 820, 80):
                draw.line((offset, 100, 900 - offset // 2, 1100), fill="black", width=8)
            image.save(path)

            result = image_preflight({"path": str(path), "role": "person"})

            self.assertEqual(result["metrics"]["width"], 900)
            self.assertEqual(result["metrics"]["height"], 1200)
            self.assertNotEqual(result["status"], "block")
            self.assertTrue(any(item["id"] == "sharpness" for item in result["checks"]))

    def test_small_image_is_blocked_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.jpg"
            Image.new("RGB", (240, 320), "gray").save(path)

            result = image_preflight({"path": str(path), "role": "garment"})

            self.assertEqual(result["status"], "block")
            dimension = next(item for item in result["checks"] if item["id"] == "dimensions")
            self.assertEqual(dimension["level"], "block")

    def test_protocol_reports_missing_file_as_structured_error(self) -> None:
        response = respond({
            "protocol_version": 1,
            "request_id": "missing",
            "op": "image.preflight",
            "payload": {"path": "does-not-exist.png", "role": "person"},
        })

        self.assertFalse(response["success"])
        self.assertEqual(response["error"]["code"], "preflight_failed")


class SafeAlbumClassificationTests(unittest.TestCase):
    def test_only_non_sensitive_coarse_groups_are_mapped(self) -> None:
        self.assertEqual(safe_category_for_imagenet_index(207), "animals")
        self.assertEqual(safe_category_for_imagenet_index(608), "outfits")
        self.assertEqual(safe_category_for_imagenet_index(963), "food")
        self.assertEqual(safe_category_for_imagenet_index(975), "nature")
        self.assertIsNone(safe_category_for_imagenet_index(480))

    def test_classifier_rejects_missing_local_weights_without_downloading(self) -> None:
        response = respond({
            "protocol_version": 1,
            "request_id": "no-model",
            "op": "album.classify_safe_batch",
            "payload": {
                "model_path": "does-not-exist.pth",
                "enabled_categories": ["animals"],
                "items": [{"id": "photo", "path": "does-not-exist.jpg"}],
            },
        })
        self.assertFalse(response["success"])
        self.assertEqual(response["error"]["code"], "album_classification_failed")


class PersonGateTests(unittest.TestCase):
    def test_multiple_candidates_block_single_person_tryon(self) -> None:
        self.assertEqual(person_presence_check(2, 1)["level"], "block")

    def test_no_candidate_requests_manual_review_instead_of_false_block(self) -> None:
        self.assertEqual(person_presence_check(0, 0)["level"], "warning")


if __name__ == "__main__":
    unittest.main()
