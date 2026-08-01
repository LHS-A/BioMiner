from __future__ import annotations

import numpy as np
import unittest

from biominer.metrics import grading_metrics


class MetricsTests(unittest.TestCase):
    def test_paper_metrics_for_perfect_predictions(self):
        probabilities = np.eye(4, dtype=np.float64)
        result = grading_metrics([0, 1, 2, 3], probabilities)
        self.assertEqual(result["overall_accuracy"], 1.0)
        self.assertEqual(result["macro_auc_ovr"], 1.0)
        self.assertEqual(result["level_accuracy"], [1.0, 1.0, 1.0, 1.0])
