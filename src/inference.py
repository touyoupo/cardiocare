"""Batch inference entrypoint with logging and input validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging

import joblib
import numpy as np
import pandas as pd

from src.constants import (
    CLINICAL_RANGES,
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    INFERENCE_LOG_PATH,
    METADATA_PATH,
    MODEL_PATH,
)


def configure_inference_logger(log_path: str | Path = INFERENCE_LOG_PATH) -> logging.Logger:
    """Configure file-based inference logging."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cardiocare.inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def validate_clinical_ranges(dataframe: pd.DataFrame) -> None:
    """Raise ValueError when inputs fall outside clinically plausible ranges."""
    for column, (lower, upper) in CLINICAL_RANGES.items():
        if column not in dataframe.columns:
            continue
        series = dataframe[column]
        if series.isna().any():
            raise ValueError(f"Column '{column}' contains missing values.")
        if (series < lower).any() or (series > upper).any():
            raise ValueError(
                f"Column '{column}' has values outside clinical range "
                f"[{lower}, {upper}]."
            )


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the serialized sklearn pipeline."""
    return joblib.load(model_path)


def load_metadata(metadata_path: str | Path = METADATA_PATH) -> dict:
    """Load model metadata such as version and training metrics."""
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))


def predict_batch(
    model,
    dataframe: pd.DataFrame,
    logger: logging.Logger | None = None,
    model_version: str = "unknown",
    ground_truth: pd.Series | None = None,
    validate_inputs: bool = True,
) -> pd.DataFrame:
    """
    Run batch inference and optionally log each batch.

    Returns a DataFrame with predictions and class probabilities.
    """
    if validate_inputs:
        validate_clinical_ranges(dataframe)
    feature_frame = dataframe[FEATURE_COLUMNS]

    predictions = model.predict(feature_frame)
    probabilities = model.predict_proba(feature_frame)

    output = feature_frame.copy()
    output["prediction"] = predictions
    output["probability_healthy"] = probabilities[:, 0]
    output["probability_disease"] = probabilities[:, 1]

    if logger is not None:
        for index, row in output.iterrows():
            actual = None
            if ground_truth is not None and index in ground_truth.index:
                actual = int(ground_truth.loc[index])
            logger.info(
                "model_version=%s | input_shape=%s | prediction=%s | "
                "prob_disease=%.4f | actual=%s",
                model_version,
                tuple(feature_frame.shape),
                int(row["prediction"]),
                float(row["probability_disease"]),
                actual,
            )
    return output


def load_input_file(input_path: str | Path) -> pd.DataFrame:
    """Load CSV or JSON batch input."""
    path = Path(input_path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="CardioCare batch inference")
    parser.add_argument(
        "--input",
        default="data/sample_batch.csv",
        help="CSV or JSON file containing patient features",
    )
    parser.add_argument("--model", default=MODEL_PATH, help="Path to trained model")
    parser.add_argument(
        "--output",
        default="artifacts/predictions.csv",
        help="Where to save prediction output",
    )
    args = parser.parse_args()

    logger = configure_inference_logger()
    model = load_model(args.model)
    metadata = load_metadata()
    input_frame = load_input_file(args.input)

    results = predict_batch(
        model=model,
        dataframe=input_frame,
        logger=logger,
        model_version=metadata.get("model_version", "unknown"),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
