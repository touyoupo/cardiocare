"""Monitoring, drift simulation, and performance degradation analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timedelta

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score, recall_score

from src.constants import (
    CLINICAL_RANGES,
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    MONITOR_OUTPUT_DIR,
    RANDOM_SEED,
)
from src.inference import configure_inference_logger, predict_batch
from src.preprocessing import binarize_target, load_raw_dataframe, split_features_target
from sklearn.model_selection import train_test_split

from src.train import build_model_pipeline, get_candidate_models


def obscure_disease_signals(
    dataframe: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Simulate covariate + label-stratified drift at monitoring time only.

    Uses held-out test labels to mimic measurement bias on disease cases — not
    used during training and therefore not training leakage.
    """
    shifted = dataframe.copy()
    mask = target == 1
    if not mask.any():
        return shifted
    shifted.loc[mask, "oldpeak"] = shifted.loc[mask, "oldpeak"] * 0.15
    shifted.loc[mask, "chol"] = (shifted.loc[mask, "chol"] - 50).clip(*CLINICAL_RANGES["chol"])
    shifted.loc[mask, "thalach"] = (shifted.loc[mask, "thalach"] + 30).clip(*CLINICAL_RANGES["thalach"])
    shifted.loc[mask, "exang"] = 0
    shifted.loc[mask, "cp"] = shifted.loc[mask, "cp"].clip(upper=2)
    return shifted


def shift_continuous_feature(
    dataframe: pd.DataFrame,
    column: str,
    mean_shift: float = 30.0,
    variance_scale: float = 1.5,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Artificially drift a continuous feature distribution."""
    rng = np.random.default_rng(random_state)
    shifted = dataframe.copy()
    base = shifted[column].astype(float)
    centered = base - base.mean()
    scaled = centered * variance_scale
    shifted[column] = scaled + base.mean() + mean_shift
    noise = rng.normal(0, 1.0, size=len(shifted))
    shifted[column] = shifted[column] + noise
    lower, upper = CLINICAL_RANGES.get(column, (-np.inf, np.inf))
    shifted[column] = shifted[column].clip(lower=lower, upper=upper)
    return shifted


def run_ks_drift_tests(
    reference: pd.DataFrame,
    shifted: pd.DataFrame,
    continuous_columns: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare reference vs shifted distributions with KS tests."""
    rows = []
    for column in continuous_columns:
        statistic, p_value = ks_2samp(
            reference[column].dropna(),
            shifted[column].dropna(),
        )
        rows.append(
            {
                "feature": column,
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drift_flag": bool(p_value < alpha),
            }
        )
    return pd.DataFrame(rows)


def evaluate_on_dataset(model, features: pd.DataFrame, target: pd.Series) -> dict:
    """Compute balanced accuracy and recall for monitoring."""
    predictions = model.predict(features)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
    }


def plot_drift_histogram(
    reference: pd.Series,
    shifted: pd.Series,
    feature_name: str,
    output_path: Path,
) -> None:
    """Plot reference vs drifted feature distributions."""
    plt.figure(figsize=(8, 5))
    sns.histplot(reference, color="steelblue", label="training", kde=True, stat="density")
    sns.histplot(shifted, color="tomato", label="drifted (test)", kde=True, stat="density")
    plt.title(f"KS Drift: {feature_name} (train vs drifted)")
    plt.xlabel(feature_name)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_metric_timeseries(
    metric_records: list[dict],
    output_path: Path,
) -> None:
    """Plot synthetic monitoring metrics over time."""
    frame = pd.DataFrame(metric_records)
    plt.figure(figsize=(9, 5))
    plt.plot(frame["timestamp"], frame["balanced_accuracy"], marker="o", label="balanced_accuracy")
    plt.plot(frame["timestamp"], frame["recall"], marker="s", label="recall")
    plt.title("Monitoring Metrics Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    """Run drift simulation and monitoring workflow."""
    output_dir = Path(MONITOR_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = binarize_target(load_raw_dataframe(DEFAULT_DATA_PATH))
    features, target = split_features_target(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=target,
    )

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        pipeline = build_model_pipeline(get_candidate_models()["random_forest"])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, model_path)
        metadata = {"model_version": "1.0"}
        Path(METADATA_PATH).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        model = pipeline
    else:
        model = joblib.load(model_path)
        metadata = json.loads(Path(METADATA_PATH).read_text(encoding="utf-8"))

    logger = configure_inference_logger(Path("logs/monitor_inference.log"))
    baseline_metrics = evaluate_on_dataset(model, X_test, y_test)

    # Shift chol to satisfy the assignment drift example and oldpeak because it is
    # one of the most influential continuous predictors in the final model.
    drift_column = "chol"
    drifted_features = shift_continuous_feature(X_test.copy(), column="chol", mean_shift=65.0)
    drifted_features = shift_continuous_feature(
        drifted_features,
        column="oldpeak",
        mean_shift=2.5,
        variance_scale=2.5,
        random_state=RANDOM_SEED + 1,
    )
    drifted_features = shift_continuous_feature(
        drifted_features,
        column="thalach",
        mean_shift=-25.0,
        variance_scale=1.8,
        random_state=RANDOM_SEED + 2,
    )
    drifted_features = obscure_disease_signals(drifted_features, y_test)
    drifted_metrics = evaluate_on_dataset(model, drifted_features, y_test)

    # PDF requirement: compare TRAINING distribution vs drifted distribution.
    continuous_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    ks_results = run_ks_drift_tests(X_train, drifted_features, continuous_columns)
    ks_results.to_csv(output_dir / "ks_drift_results.csv", index=False)

    plot_drift_histogram(
        X_train[drift_column],
        drifted_features[drift_column],
        drift_column,
        output_dir / f"drift_hist_{drift_column}.png",
    )

    predict_batch(
        model=model,
        dataframe=drifted_features.head(5),
        logger=logger,
        model_version=metadata.get("model_version", "unknown"),
        ground_truth=y_test.head(5),
        validate_inputs=False,
    )

    start_time = datetime(2026, 1, 1, 8, 0, 0)
    metric_records = []
    for step in range(6):
        decay = step * 0.02
        metric_records.append(
            {
                "timestamp": start_time + timedelta(days=step),
                "balanced_accuracy": max(
                    0.0, baseline_metrics["balanced_accuracy"] - decay
                ),
                "recall": max(0.0, baseline_metrics["recall"] - decay - 0.01),
            }
        )
    plot_metric_timeseries(metric_records, output_dir / "metric_timeseries.png")

    summary = {
        "baseline_metrics": baseline_metrics,
        "drifted_metrics": drifted_metrics,
        "recall_drop": baseline_metrics["recall"] - drifted_metrics["recall"],
        "balanced_accuracy_drop": (
            baseline_metrics["balanced_accuracy"] - drifted_metrics["balanced_accuracy"]
        ),
        "flagged_features": ks_results.loc[ks_results["drift_flag"], "feature"].tolist(),
        "ks_reference": "training_set",
        "ks_comparison": "drifted_test_set",
        "retraining_policy": (
            "Trigger retraining when KS p-value < 0.05 AND recall drops by more than 5%, "
            "with cardiologist review before deployment."
        ),
    }
    summary_path = output_dir / "monitor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Baseline metrics:", baseline_metrics)
    print("Drifted metrics:", drifted_metrics)
    print("KS drift results:\n", ks_results)
    print("Flagged features:", summary["flagged_features"])
    print(f"Saved monitoring artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
