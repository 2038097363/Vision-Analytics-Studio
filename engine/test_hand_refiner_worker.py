import tempfile
import unittest
from pathlib import Path

from engine.hand_refiner_worker import _hand_geometry, model_status, respond


class HandRefinerWorkerTests(unittest.TestCase):
    def test_hand_geometry_extends_beyond_wrist_with_a_broad_palm_region(self) -> None:
        crop_box, mask = _hand_geometry(
            (576, 768),
            [[0.42, 0.33], [0.39, 0.60], [0.49, 0.80]],
        )
        self.assertLessEqual(crop_box[1], 490)
        self.assertEqual(crop_box[3], 768)
        self.assertGreater(mask.getbbox()[2], int(0.49 * 576))
        self.assertGreater(mask.histogram()[255], 15000)

    def test_incomplete_install_is_reported_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = model_status({
                "runtime_root": str(root / "runtime"),
                "weights_root": str(root / "weights"),
            })
        self.assertFalse(result["installed"])
        self.assertFalse(result["ready"])
        self.assertFalse(result["automatic_download"])
        self.assertEqual(result["release_status"], "evaluation-required")

    def test_unknown_operation_fails_closed(self) -> None:
        response = respond({
            "protocol_version": 1,
            "request_id": "test",
            "op": "hand_refiner.repair",
            "payload": {},
        })
        self.assertFalse(response["success"])
        self.assertEqual(response["error"]["code"], "hand_refiner_failed")


if __name__ == "__main__":
    unittest.main()
