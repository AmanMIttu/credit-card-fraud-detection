"""Focused tests for V2 dashboard validation and prediction helpers."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from app import _load_assets, predict_transactions, predict_csv_bytes

ROOT = Path(__file__).resolve().parent

class DashboardPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metadata = _load_assets()
        cls.columns = cls.metadata["feature_columns"]
        cls.row = {column: 0.0 for column in cls.columns}
        cls.row["Time"] = 120.0
        cls.row["Amount"] = 25.0

    def test_saved_pipeline_and_threshold_load(self):
        self.assertEqual(self.metadata["selected_model"], "Random Forest")
        self.assertAlmostEqual(self.metadata["selected_threshold"], 0.3397559744256)
        self.assertIn("preprocess", self.model.named_steps)
        self.assertIn("model", self.model.named_steps)

    def test_single_prediction_and_probability(self):
        result, notices = predict_transactions(pd.DataFrame([self.row]), self.model, self.metadata)
        self.assertEqual(len(result), 1)
        self.assertTrue(0 <= result.loc[0, "Fraud Probability"] <= 1)
        expected = "Fraud" if result.loc[0, "Fraud Probability"] >= self.metadata["selected_threshold"] else "Legitimate"
        self.assertEqual(result.loc[0, "Decision"], expected)
        self.assertEqual(notices, [])

    def test_batch_csv_prediction(self):
        rows = pd.DataFrame([self.row, self.row])
        result, _ = predict_csv_bytes(rows.to_csv(index=False).encode("utf-8"), self.model, self.metadata)
        self.assertEqual(len(result), 2)
        self.assertIn("Fraud Probability", result.columns)
        self.assertIn("Decision", result.columns)

    def test_missing_features_are_rejected(self):
        bad = pd.DataFrame([{k:v for k,v in self.row.items() if k != "V1"}])
        with self.assertRaisesRegex(ValueError, "Missing required feature columns.*V1"):
            predict_transactions(bad, self.model, self.metadata)

    def test_nonnumeric_and_missing_values_are_rejected(self):
        bad = pd.DataFrame([{**self.row, "V2": "not-a-number"}])
        with self.assertRaisesRegex(ValueError, "V2.*non-numeric"):
            predict_transactions(bad, self.model, self.metadata)
        bad = pd.DataFrame([self.row]); bad.loc[0,"V3"] = np.nan
        with self.assertRaisesRegex(ValueError, "V3.*missing"):
            predict_transactions(bad, self.model, self.metadata)

    def test_nonfinite_and_negative_amount_are_rejected(self):
        bad = pd.DataFrame([self.row]); bad.loc[0,"V4"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            predict_transactions(bad, self.model, self.metadata)
        bad = pd.DataFrame([self.row]); bad.loc[0,"Amount"] = -1
        with self.assertRaisesRegex(ValueError, "Amount.*negative"):
            predict_transactions(bad, self.model, self.metadata)

    def test_target_column_is_ignored_and_not_returned(self):
        row = dict(self.row, Class=1)
        result, notices = predict_transactions(pd.DataFrame([row]), self.model, self.metadata)
        self.assertNotIn("Class", result.columns)
        self.assertTrue(any("Class" in notice for notice in notices))

if __name__ == "__main__":
    unittest.main()
