"""Generate EDA figures for the final report (boxplot + class distribution)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RANDOM_SEED, TARGET_COLUMN
from src.preprocessing import binarize_target, load_raw_dataframe

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "report"
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = binarize_target(load_raw_dataframe())

    # Figure: target class distribution
    plt.figure(figsize=(6, 4))
    counts = df[TARGET_COLUMN].value_counts(normalize=True).sort_index()
    ax = sns.barplot(
        x=["Healthy (0)", "Disease (1)"],
        y=[counts.get(0, 0), counts.get(1, 0)],
        hue=["Healthy (0)", "Disease (1)"],
        palette=["#4C78A8", "#F58518"],
        legend=False,
    )
    ax.set_title("Target Class Distribution (EDA)")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    for index, value in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(index, value + 0.02, f"{value:.2%}", ha="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_target_distribution.png", dpi=150)
    plt.close()

    # Figure: continuous feature boxplots
    fig, axes = plt.subplots(1, len(CONTINUOUS), figsize=(14, 4))
    for axis, feature in zip(axes, CONTINUOUS):
        sns.boxplot(y=df[feature], ax=axis, color="#72B7B2")
        axis.set_title(feature)
        axis.set_xlabel("")
    fig.suptitle("Outlier Inspection — Continuous Features (EDA)", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved EDA figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
