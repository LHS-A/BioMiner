from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from biominer.data import FUNCTIONAL_GRADING_QUESTION, load_text_samples


class DataTests(unittest.TestCase):
    def test_structured_examples_render_to_paper_narrative(self):
        payload = [
            {
                "id": "example",
                "age": 40,
                "gender": "female",
                "systemic_comorbidities": [],
                "duration_years": 0,
                "morphometrics": {
                    "langerhans_cell_density": 10,
                    "mean_field_area": 70,
                    "mean_dendrite_tips": 1,
                    "mean_cell_perimeter": 35,
                    "mean_cell_aspect_ratio": 1.2,
                    "mean_cell_solidity": 0.9,
                    "nerve_fiber_length": 20,
                    "nerve_fiber_density": 30,
                    "nerve_branch_point_density": 80,
                    "nerve_curvature_index": 1.1,
                    "nerve_fractal_dimension": 1.4,
                    "nerve_fiber_area": 0.07,
                },
                "nerve_label": 0,
                "cell_label": 1,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            sample = load_text_samples(path)[0]
        self.assertIn("Langerhans cell density = 10 cells/mm2", sample["input"])
        self.assertTrue(sample["input"].endswith(FUNCTIONAL_GRADING_QUESTION))

    def test_invalid_grade_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps([{"input": "text", "nerve_label": 4, "cell_label": 0}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nerve_label"):
                load_text_samples(path)
