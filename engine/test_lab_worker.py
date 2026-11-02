import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from engine.lab_worker import respond, segment, validate_parameters


class LabWorkerTests(unittest.TestCase):
    def test_parameter_limits(self) -> None:
        with self.assertRaises(ValueError):
            validate_parameters({"threshold_scale": 2})
        with self.assertRaises(ValueError):
            validate_parameters({"min_area": 1})

    def test_two_separate_nuclei_are_measured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "nuclei.png"
            output_path = root / "overlay.png"
            image = Image.new("L", (256, 256), 3)
            drawing = ImageDraw.Draw(image)
            drawing.ellipse((35, 60, 95, 120), fill=245)
            drawing.ellipse((150, 135, 220, 205), fill=220)
            image.save(image_path)
            result = segment({
                "image_path": str(image_path), "output_path": str(output_path),
                "threshold_scale": 1.0, "min_area": 40, "exclude_edges": True,
                "scale_um_per_px": 0.5,
            })
            output_created = output_path.is_file()
            self.assertEqual(result["summary"]["count"], 2)
            self.assertEqual(result["summary"]["unit"], "μm")
            self.assertTrue(output_created)

    def test_invalid_operation_is_structured(self) -> None:
        response = respond({"protocol_version": 1, "request_id": "bad", "op": "lab.guess", "payload": {}})
        self.assertFalse(response["success"])
        self.assertEqual(response["error"]["code"], "lab_failed")


if __name__ == "__main__":
    unittest.main()
