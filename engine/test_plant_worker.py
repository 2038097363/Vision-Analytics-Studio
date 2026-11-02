import tempfile
import unittest
from pathlib import Path

from engine.plant_worker import candidate_decision, model_status, respond


class PlantWorkerTests(unittest.TestCase):
    def test_missing_weights_are_reported_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = model_status({"weights_dir": directory})
        self.assertFalse(result["installed"])
        self.assertFalse(result["ready"])
        self.assertEqual(result["supported_taxa"], 20)

    def test_scores_never_become_automatic_confirmation(self) -> None:
        self.assertEqual(candidate_decision([0.99, 0.005, 0.005]), "uncertain")
        self.assertEqual(candidate_decision([]), "unsupported")

    def test_invalid_organ_fails_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "plant.png"
            image.write_bytes(b"not needed for this validation")
            response = respond({
                "protocol_version": 1,
                "request_id": "invalid-organ",
                "op": "plant.identify",
                "payload": {"image_path": str(image), "weights_dir": directory, "organ": "root"},
            })
        self.assertFalse(response["success"])
        self.assertIn("部位", response["error"]["message"])


if __name__ == "__main__":
    unittest.main()
