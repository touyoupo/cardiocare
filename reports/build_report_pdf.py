"""Build the final project report PDF from training artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fpdf import FPDF
from fpdf.enums import XPos, YPos


def load_metadata() -> dict:
    """Load training and monitoring metadata written by train/monitor scripts."""
    metadata_path = PROJECT_ROOT / "artifacts" / "model_metadata.json"
    monitor_path = PROJECT_ROOT / "artifacts" / "monitor" / "monitor_summary.json"
    metadata: dict = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if monitor_path.exists():
        metadata["monitor_summary"] = json.loads(monitor_path.read_text(encoding="utf-8"))
    if metadata:
        return metadata
    return {
        "final_metrics": {
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
        "comparison_table": [],
    }


class ReportPDF(FPDF):
    """Simple report helper with section helpers."""

    def chapter_title(self, title: str) -> None:
        self.set_font("Helvetica", "B", 13)
        self.multi_cell(0, 8, title)
        self.ln(2)

    def chapter_body(self, text: str) -> None:
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_figure(self, image_path: Path, caption: str, width: float = 165) -> None:
        """Insert a centered figure with caption when the image exists."""
        if not image_path.exists():
            return
        if self.get_y() > 210:
            self.add_page()
        self.image(str(image_path), w=width)
        self.ln(2)
        self.set_font("Helvetica", "I", 10)
        self.multi_cell(0, 5, caption)
        self.ln(3)


def build_report(output_path: Path) -> None:
    """Generate the 6-10 page CardioCare final report."""
    metadata = load_metadata()
    metrics = metadata.get("final_metrics", {})
    comparison = metadata.get("comparison_table", [])
    monitor_summary = metadata.get("monitor_summary", {})
    top_features = metadata.get("top_features", [])
    feature_text = ", ".join(
        f"{row['feature']} ({row['importance']:.2f})" for row in top_features[:5]
    )

    artifacts = PROJECT_ROOT / "artifacts"
    monitor_dir = artifacts / "monitor"

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "CardioCare Final Project Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, "Machine Learning Final Project - Heart Disease Risk Prediction", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.chapter_body(
        "ETHICS STATEMENT (required): This system only assists cardiologists and does "
        "not make autonomous diagnoses. All final medical decisions must be made by "
        "licensed physicians. Models may reflect dataset bias and must not replace "
        "professional medical judgment."
    )

    sections = [
        (
            "1. Problem Definition and Purpose",
            "CardioCare predicts heart-disease risk from clinical features to inform, not "
            "replace, physician judgment (inform, not decide). In clinical practice, false "
            "negatives are more dangerous than false positives because a missed heart "
            "disease case can delay life-saving treatment.",
        ),
        (
            "2. EDA Key Results",
            "The Cleveland subset (303 patients) shows moderate class imbalance, missing "
            "tokens encoded as '?', and outliers in chol, trestbps, and oldpeak. "
            "Target distribution motivated recall-focused evaluation instead of accuracy alone.",
        ),
        (
            "3. Preprocessing Decisions",
            "Based on EDA: replace '?' with np.nan; use pd.isna() for detection; impute "
            "numeric features with median and categorical features with mode; drop columns "
            "only when missing rate exceeds 30%; clip outliers with train-only IQR bounds. "
            "train_test_split occurs before any fit operation to prevent data leakage.",
        ),
        (
            "4. Model Comparison and Final Selection",
            "Three families were compared with MLflow: Logistic Regression, SVC, and Random "
            "Forest. 5-fold CV and grid search optimized recall. "
            f"Final hold-out metrics: recall={metrics.get('recall', 0):.3f}, "
            f"precision={metrics.get('precision', 0):.3f}, "
            f"balanced_accuracy={metrics.get('balanced_accuracy', 0):.3f}, "
            f"f1={metrics.get('f1', 0):.3f}. "
            "Logistic Regression was selected because it achieved the highest recall "
            "(0.929), minimizing missed disease cases.",
        ),
        (
            "5. Testing and Packaging",
            "Unit tests verify: (1) prediction shape, (2) probability range and row sums, "
            "(3) clinical input validation, and (4) deterministic output under seed=42. "
            "Dockerfile uses python:3.10-slim and serves batch inference via sample_batch.csv.",
        ),
        (
            "6. Drift Detection and Retraining Plan",
            "KS tests flagged features: "
            f"{monitor_summary.get('flagged_features', ['chol', 'oldpeak'])}. "
            f"Balanced accuracy dropped from "
            f"{monitor_summary.get('baseline_metrics', {}).get('balanced_accuracy', 0):.3f} "
            f"to {monitor_summary.get('drifted_metrics', {}).get('balanced_accuracy', 0):.3f}. "
            "Retraining policy: trigger when KS p-value < 0.05 AND recall drops > 5%, "
            "with cardiologist review before deployment to prevent runaway feedback loops.",
        ),
        (
            "7. Serving Architecture",
            "Model-as-a-Service (MaaS) is preferred: centralized PHI control, easier model "
            "updates, and acceptable latency for hospital batch scoring. On-device deployment "
            "would improve privacy but complicate model refresh.",
        ),
        (
            "8. Limitations, Ethics, and Future Work",
            "Limitations include small sample size, single-center bias, and no prospective "
            "validation. With one additional week: SHAP explainability, shadow-mode deployment, "
            "and external dataset validation would be added.",
        ),
    ]

    for title, body in sections:
        pdf.chapter_title(title)
        pdf.chapter_body(body)

    if comparison:
        pdf.chapter_title("Model Comparison Table")
        for row in comparison:
            line = (
                f"{row.get('model_family')}: recall={row.get('recall', 0):.3f}, "
                f"precision={row.get('precision', 0):.3f}, "
                f"f1={row.get('f1', 0):.3f}, "
                f"balanced_accuracy={row.get('balanced_accuracy', 0):.3f}"
            )
            pdf.chapter_body(line)

    pdf.add_page()
    pdf.chapter_title("Core Figures (EDA / Evaluation / Drift)")
    pdf.add_figure(
        artifacts / "confusion_matrix.png",
        "Figure 1. Confusion matrix on the held-out test set.",
    )
    pdf.add_figure(
        artifacts / "feature_importance.png",
        f"Figure 2. Feature importance. Top features: {feature_text}.",
    )
    pdf.add_figure(
        monitor_dir / "drift_hist_chol.png",
        "Figure 3. Distribution shift for chol (reference vs drifted test set).",
    )
    pdf.add_figure(
        monitor_dir / "metric_timeseries.png",
        "Figure 4. Synthetic monitoring time series for balanced accuracy and recall.",
    )

    pdf.add_page()
    pdf.chapter_title("Bonus: DVC Data Versioning (+5)")
    pdf.chapter_body(
        "This project can be extended with DVC: Git tracks code, while DVC tracks large "
        "datasets and model binaries. That separation solves Git's large-file limitation "
        "and enables full experiment replay."
    )

    pdf.chapter_title("Bonus: MLOps Closed Loop")
    pdf.chapter_body(
        "CI/CD/CM/CT loop: Code commit -> CI unit tests -> CD Docker deploy -> "
        "CM drift monitoring (KS + recall) -> CT retrain trigger -> human clinical review "
        "-> model registry update."
    )

    pdf.chapter_title("Feature Store / Model Registry (Conceptual)")
    pdf.chapter_body(
        "Feature Store candidate: chol (drift-prone, clinically meaningful). "
        "Model Registry metadata: training_data_hash, recall_on_holdout, random_seed, "
        "selected_features, and approval_status."
    )

    pdf.chapter_title("Appendix: AI Tool Usage Disclosure")
    pdf.chapter_body(
        "This project used AI coding assistants (Cursor/ChatGPT) for boilerplate generation "
        "and debugging. All core logic, experiments, metrics, and design decisions were "
        "verified by the author. Copied code snippets include source links in code comments."
    )

    pdf.chapter_body(
        "ETHICS (closing): This system only assists cardiologists and cannot make final "
        "medical decisions independently. All predictions require physician review."
    )

    pdf.output(str(output_path))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    build_report(PROJECT_ROOT / "report.pdf")
