"""Optional helper: run full local workflow beyond README grading steps."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_step(command: list[str], description: str) -> None:
    """Execute one pipeline step and fail fast on errors."""
    print(f"\n=== {description} ===")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    """Train, test, monitor, infer, and rebuild report (see README for grading steps)."""
    run_step([PYTHON, "src/train.py"], "Train model")
    run_step(
        [PYTHON, "-m", "unittest", "discover", "-s", "tests", "-v"],
        "Run unit tests",
    )
    run_step([PYTHON, "src/monitor.py"], "Run drift monitor")
    run_step(
        [PYTHON, "src/inference.py", "--input", "data/sample_batch.csv"],
        "Run batch inference",
    )
    run_step([PYTHON, "reports/generate_eda_figures.py"], "Generate EDA figures")
    run_step([PYTHON, "reports/build_report_pdf.py"], "Build report.pdf")
    print("\nPipeline complete. See report.pdf, artifacts/, and mlruns/.")


if __name__ == "__main__":
    main()
