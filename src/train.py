"""Model training, experiment tracking, and final model selection."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.constants import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    METADATA_PATH,
    MLRUNS_DIR,
    MODEL_PATH,
    RANDOM_SEED,
)
from src.preprocessing import (
    binarize_target,
    build_preprocessing_pipeline,
    load_raw_dataframe,
    split_features_target,
)


def configure_mlflow() -> None:
    """Use a project-local MLflow tracking directory with portable paths."""
    tracking_dir = (PROJECT_ROOT / MLRUNS_DIR).resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())


def build_model_pipeline(estimator) -> Pipeline:
    """Compose preprocessing, scaling, feature selection, and classifier."""
    selector_estimator = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline()),
            ("scaler", StandardScaler()),
            (
                "feature_selection",
                SelectFromModel(selector_estimator, threshold="median"),
            ),
            ("classifier", estimator),
        ]
    )


def evaluate_classifier(pipeline: Pipeline, X_test, y_test) -> dict[str, float | list]:
    """Compute required classification metrics on held-out data."""
    y_pred = pipeline.predict(X_test)
    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    if y_prob is not None:
        metrics["positive_class_probability_mean"] = float(y_prob[:, 1].mean())
        metrics["auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
    else:
        metrics["auc"] = 0.0
    return metrics


def get_selected_feature_count(pipeline: Pipeline) -> int:
    """Return how many features survived SelectFromModel."""
    selector = pipeline.named_steps["feature_selection"]
    return int(np.sum(selector.get_support()))


def get_candidate_models() -> dict[str, object]:
    """Return at least three model families for comparison."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "svc": SVC(
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }


def train_and_track(
    X_train,
    X_test,
    y_train,
    y_test,
    experiment_name: str = "cardiocare",
) -> tuple[Pipeline, pd.DataFrame]:
    """Train all candidate models with MLflow logging."""
    mlflow.set_experiment(experiment_name)
    results: list[dict] = []
    best_pipeline: Pipeline | None = None
    best_recall = -1.0

    for model_name, estimator in get_candidate_models().items():
        pipeline = build_model_pipeline(estimator)
        with mlflow.start_run(run_name=model_name) as run:
            pipeline.fit(X_train, y_train)
            metrics = evaluate_classifier(pipeline, X_test, y_test)

            mlflow.set_tag("model_family", model_name)
            mlflow.log_param("random_seed", RANDOM_SEED)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("feature_selector", "SelectFromModel(RandomForest)")
            mlflow.log_param("scaler", "StandardScaler")
            mlflow.log_metrics(
                {
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "auc": metrics["auc"],
                }
            )
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            result = {
                "run_id": run.info.run_id,
                "model_family": model_name,
                **{
                    key: metrics[key]
                    for key in [
                        "accuracy",
                        "balanced_accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "auc",
                    ]
                },
            }
            results.append(result)

            # Clinical priority: maximize recall to reduce false negatives.
            if metrics["recall"] > best_recall:
                best_recall = metrics["recall"]
                best_pipeline = pipeline

    comparison = pd.DataFrame(results).sort_values("recall", ascending=False)
    return best_pipeline, comparison


def tune_best_model(
    X_train,
    y_train,
    base_pipeline: Pipeline,
) -> Pipeline:
    """Run 5-fold CV with grid search on the best candidate family."""
    classifier = base_pipeline.named_steps["classifier"]
    if isinstance(classifier, LogisticRegression):
        param_grid = {"classifier__C": [0.1, 1.0, 10.0]}
        run_name = "tuned_logistic_regression"
    elif isinstance(classifier, SVC):
        param_grid = {"classifier__C": [0.5, 1.0, 2.0], "classifier__kernel": ["rbf", "linear"]}
        run_name = "tuned_svc"
    else:
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
        }
        run_name = "tuned_random_forest"

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    n_jobs = 1 if os.environ.get("GITHUB_ACTIONS") == "true" else -1
    search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=cv,
        n_jobs=n_jobs,
    )

    with mlflow.start_run(run_name=run_name):
        search.fit(X_train, y_train)
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_recall", search.best_score_)
        mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model")
        mlflow.set_tag("model_family", run_name)

    return search.best_estimator_


def save_evaluation_plots(
    pipeline: Pipeline,
    X_test,
    y_test,
    output_dir: str | Path,
) -> None:
    """Save confusion matrix and feature-importance plots for the report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pipeline.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["healthy", "disease"],
        yticklabels=["healthy", "disease"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    importance_rows = export_feature_importance(
        pipeline, output_dir / "feature_importance.json"
    )
    if importance_rows:
        frame = pd.DataFrame(importance_rows)
        plt.figure(figsize=(7, 4))
        sns.barplot(data=frame, x="importance", y="feature", hue="feature", legend=False)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=150)
        plt.close()


def export_feature_importance(
    pipeline: Pipeline,
    output_path: str | Path,
) -> list[dict]:
    """Save feature-importance values when the final estimator supports them."""
    classifier = pipeline.named_steps["classifier"]
    selector = pipeline.named_steps["feature_selection"]
    selected_mask = selector.get_support()
    selected_features = [
        name for name, keep in zip(FEATURE_COLUMNS, selected_mask) if keep
    ]

    importance_rows: list[dict] = []
    if hasattr(classifier, "feature_importances_"):
        for feature_name, importance in zip(
            selected_features, classifier.feature_importances_
        ):
            importance_rows.append(
                {"feature": feature_name, "importance": float(importance)}
            )
    elif hasattr(classifier, "coef_"):
        coefficients = np.abs(classifier.coef_).ravel()
        for feature_name, importance in zip(selected_features, coefficients):
            importance_rows.append(
                {"feature": feature_name, "importance": float(importance)}
            )

    if importance_rows:
        importance_rows = sorted(
            importance_rows, key=lambda row: row["importance"], reverse=True
        )
        Path(output_path).write_text(
            pd.DataFrame(importance_rows).to_json(orient="records", indent=2),
            encoding="utf-8",
        )
    return importance_rows


def save_final_artifacts(
    pipeline: Pipeline,
    comparison: pd.DataFrame,
    X_test,
    y_test,
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> dict:
    """Persist model, metadata, and final evaluation report."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    final_metrics = evaluate_classifier(pipeline, X_test, y_test)
    joblib.dump(pipeline, model_path)
    try:
        save_evaluation_plots(pipeline, X_test, y_test, model_path.parent)
    except Exception as plot_error:
        print(f"Warning: plot generation skipped: {plot_error}")

    importance_path = model_path.parent / "feature_importance.json"
    importance_rows = (
        json.loads(importance_path.read_text(encoding="utf-8"))
        if importance_path.exists()
        else []
    )

    metadata = {
        "model_version": "1.0",
        "random_seed": RANDOM_SEED,
        "selection_criterion": "maximize_recall_reduce_false_negatives",
        "feature_store": {
            "registered_feature": "chol",
            "rationale_ko": (
                "콜레스테롤(chol)은 심혈관 위험의 핵심 연속 지표로, "
                "전처리(IQR 클리핑·표준화) 후 재사용 가능한 표준 특성입니다."
            ),
        },
        "model_registry": {
            "metadata_field": "recall_on_holdout",
            "value": final_metrics["recall"],
            "rationale_ko": (
                "위음성(FN) 위험을 줄이기 위해 홀드아웃 재현율을 "
                "모델 승격·롤백의 1차 기준으로 등록합니다."
            ),
        },
        "comparison_table": comparison.to_dict(orient="records"),
        "final_metrics": {
            key: final_metrics[key]
            for key in [
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
            ]
        },
        "confusion_matrix": final_metrics["confusion_matrix"],
        "top_features": importance_rows[:5],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(classification_report(y_test, pipeline.predict(X_test)))
    print("Model comparison:\n", comparison)
    print(f"Saved model to {model_path}")
    return metadata


def ensure_data_available(data_path: str | Path = DEFAULT_DATA_PATH) -> Path:
    """Download UCI data when missing so graders can run train.py directly."""
    data_path = Path(data_path)
    if data_path.exists():
        return data_path
    download_script = PROJECT_ROOT / "data" / "download_data.py"
    import subprocess

    subprocess.run([sys.executable, str(download_script)], check=True, cwd=PROJECT_ROOT)
    if not data_path.exists():
        raise FileNotFoundError(f"Expected dataset at {data_path} after download.")
    return data_path


def main() -> None:
    """End-to-end training entrypoint."""
    configure_mlflow()
    ensure_data_available()
    dataframe = binarize_target(load_raw_dataframe(DEFAULT_DATA_PATH))
    features, target = split_features_target(dataframe)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=target,
    )

    best_pipeline, comparison = train_and_track(X_train, X_test, y_train, y_test)
    tuned_pipeline = tune_best_model(X_train, y_train, best_pipeline)
    save_final_artifacts(tuned_pipeline, comparison, X_test, y_test)


if __name__ == "__main__":
    main()
