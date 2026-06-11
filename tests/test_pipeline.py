"""Unit tests for CardioCare ML pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.constants import CLINICAL_RANGES, FEATURE_COLUMNS, RANDOM_SEED
from src.inference import validate_clinical_ranges
from src.preprocessing import build_preprocessing_pipeline, load_raw_dataframe
from src.train import build_model_pipeline, evaluate_classifier


class CardioCarePipelineTests(unittest.TestCase):
    """Validate inference behavior and reproducibility."""

    @classmethod
    def setUpClass(cls) -> None:
        data_path = Path("data/heart_disease.csv")
        if data_path.exists():
            dataframe = load_raw_dataframe(data_path)
            cls.sample_frame = dataframe[FEATURE_COLUMNS].head(20).copy()
            cls.sample_target = (dataframe["target"].head(20) > 0).astype(int)
        else:
            cls.sample_frame = cls._build_synthetic_frame(20)
            cls.sample_target = pd.Series([0, 1] * 10)

        cls.model = build_model_pipeline(
            RandomForestClassifier(
                n_estimators=50,
                random_state=RANDOM_SEED,
                class_weight="balanced",
            )
        )
        cls.model.fit(cls.sample_frame, cls.sample_target)

    @staticmethod
    def _build_synthetic_frame(rows: int) -> pd.DataFrame:
        rng = np.random.default_rng(RANDOM_SEED)
        data = {
            "age": rng.integers(29, 80, size=rows),
            "sex": rng.integers(0, 2, size=rows),
            "cp": rng.integers(1, 5, size=rows),
            "trestbps": rng.integers(90, 180, size=rows),
            "chol": rng.integers(150, 400, size=rows),
            "fbs": rng.integers(0, 2, size=rows),
            "restecg": rng.integers(0, 3, size=rows),
            "thalach": rng.integers(80, 200, size=rows),
            "exang": rng.integers(0, 2, size=rows),
            "oldpeak": rng.uniform(0, 4, size=rows),
            "slope": rng.integers(1, 4, size=rows),
            "ca": rng.integers(0, 4, size=rows),
            "thal": rng.choice([3, 6, 7], size=rows),
        }
        return pd.DataFrame(data)

    def test_prediction_shape_matches_input(self) -> None:
        predictions = self.model.predict(self.sample_frame)
        self.assertEqual(predictions.shape[0], self.sample_frame.shape[0])

    def test_prediction_probabilities_are_valid(self) -> None:
        probabilities = self.model.predict_proba(self.sample_frame)
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        row_sums = probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5, atol=1e-5)

    def test_clinical_range_validation_rejects_invalid_chol(self) -> None:
        invalid_frame = self.sample_frame.copy()
        invalid_frame.loc[0, "chol"] = 999
        with self.assertRaises(ValueError):
            validate_clinical_ranges(invalid_frame)

    def test_clinical_range_validation_accepts_valid_input(self) -> None:
        valid_frame = self.sample_frame.copy()
        for column, (lower, upper) in CLINICAL_RANGES.items():
            valid_frame[column] = valid_frame[column].clip(lower=lower, upper=upper)
        validate_clinical_ranges(valid_frame)

    def test_pipeline_is_deterministic_with_fixed_seed(self) -> None:
        model_a = build_model_pipeline(
            RandomForestClassifier(
                n_estimators=50,
                random_state=RANDOM_SEED,
                class_weight="balanced",
            )
        )
        model_b = build_model_pipeline(
            RandomForestClassifier(
                n_estimators=50,
                random_state=RANDOM_SEED,
                class_weight="balanced",
            )
        )
        model_a.fit(self.sample_frame, self.sample_target)
        model_b.fit(self.sample_frame, self.sample_target)

        predictions_a = model_a.predict(self.sample_frame)
        predictions_b = model_b.predict(self.sample_frame)
        np.testing.assert_array_equal(predictions_a, predictions_b)

    def test_evaluate_classifier_returns_required_metrics(self) -> None:
        metrics = evaluate_classifier(self.model, self.sample_frame, self.sample_target)
        for key in [
            "accuracy",
            "auc",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "confusion_matrix",
        ]:
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
