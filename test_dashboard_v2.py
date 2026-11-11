"""Focused tests for V2 dashboard validation and prediction helpers."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from app import _load_assets, predict_transactions, predict_csv_bytes
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "creditcard.csv"

class DashboardPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metadata = _load_assets()
        cls.columns = cls.metadata["feature_columns"]
        cls.row = {column: 0.0 for column in cls.columns}
        cls.row["Time"] = 120.0
        cls.row["Amount"] = 25.0
        cls.demo_rows = {}
        if DATASET_PATH.is_file():
            selected_columns = [*cls.columns, "Class"]
            with pd.read_csv(DATASET_PATH, usecols=selected_columns, chunksize=25_000) as reader:
                for chunk in reader:
                    chunk = chunk.loc[:, selected_columns]
                    for class_value, label in ((0, "legitimate"), (1, "fraudulent")):
                        if label in cls.demo_rows:
                            continue
                        matches = chunk.loc[chunk["Class"] == class_value]
                        if not matches.empty:
                            cls.demo_rows[label] = {
                                column: float(matches.iloc[0][column]) for column in cls.columns
                            }
                    if len(cls.demo_rows) == 2:
                        break

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


    def _assert_demo_load_and_score(self, button_key, example_name):
        self.assertIn(example_name, self.demo_rows)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        self.assertEqual([], [error.message for error in at.exception])
        at.button(key=button_key).click().run(timeout=90)
        self.assertEqual([], [error.message for error in at.exception])

        loaded_notice = " ".join(
            [alert.value for alert in at.success] + [alert.value for alert in at.info]
        )
        self.assertIn("Demonstration data loaded", loaded_notice)
        self.assertIn(example_name, loaded_notice)
        self.assertEqual(30, len(at.number_input))
        self.assertNotIn("Class", self.columns)

        expected = self.demo_rows[example_name]
        for feature in self.columns:
            self.assertAlmostEqual(
                expected[feature],
                float(at.get_by_key(f"single_{feature}").value),
                places=6,
                msg=f"Loaded value mismatch for {feature}",
            )

        # Demonstrate that dataset examples remain editable before submission.
        edited_feature = "V1"
        edited_value = expected[edited_feature] + 0.1
        at.number_input(key=f"single_{edited_feature}").set_value(edited_value)
        at.form_submit_button(key="score_single_transaction").click().run(timeout=90)
        self.assertEqual([], [error.message for error in at.exception])
        result_tables = [element.value for element in at.dataframe if isinstance(element.value, pd.DataFrame)]
        prediction = next(table for table in result_tables if "Fraud Probability" in table.columns)
        self.assertAlmostEqual(edited_value, float(prediction.loc[0, edited_feature]), places=6)
        self.assertNotIn("Class", prediction.columns)
        self.assertTrue(0.0 <= float(prediction.loc[0, "Fraud Probability"]) <= 1.0)
        self.assertIn(prediction.loc[0, "Decision"], {"Fraud", "Legitimate"})

    @unittest.skipUnless(DATASET_PATH.is_file(), "Requires the locally provided creditcard.csv dataset")
    def test_load_and_score_actual_legitimate_example(self):
        self._assert_demo_load_and_score("load_legitimate_example", "legitimate")

    @unittest.skipUnless(DATASET_PATH.is_file(), "Requires the locally provided creditcard.csv dataset")
    def test_load_and_score_actual_fraudulent_example(self):
        self._assert_demo_load_and_score("load_fraudulent_example", "fraudulent")

    @unittest.skipUnless(DATASET_PATH.is_file(), "Requires the locally provided creditcard.csv dataset")
    def test_reset_form_restores_original_zero_defaults(self):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        at.button(key="load_fraudulent_example").click().run(timeout=90)
        self.assertEqual([], [error.message for error in at.exception])
        at.button(key="reset_single_form").click().run(timeout=90)
        self.assertEqual([], [error.message for error in at.exception])
        for feature in self.columns:
            self.assertEqual(0.0, float(at.get_by_key(f"single_{feature}").value), feature)
        reset_notice = " ".join(alert.value for alert in at.info)
        self.assertIn("Form reset to its original zero defaults", reset_notice)

    def test_target_column_is_ignored_and_not_returned(self):
        row = dict(self.row, Class=1)
        result, notices = predict_transactions(pd.DataFrame([row]), self.model, self.metadata)
        self.assertNotIn("Class", result.columns)
        self.assertTrue(any("Class" in notice for notice in notices))

if __name__ == "__main__":
    unittest.main()
