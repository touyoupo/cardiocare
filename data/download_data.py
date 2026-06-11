"""Deterministic download script for the UCI Cleveland Heart Disease dataset.

Dataset reference: https://archive.ics.uci.edu/dataset/45/heart+disease
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import DATA_URL, DEFAULT_DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN


def download_heart_disease(output_path: str | Path = DEFAULT_DATA_PATH) -> Path:
    """
    Download and persist the Cleveland heart disease subset.

    The UCI file uses '?' for missing values; we keep them as strings so that
    downstream preprocessing can replace them with np.nan explicitly.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    column_names = FEATURE_COLUMNS + [TARGET_COLUMN]
    dataframe = pd.read_csv(
        DATA_URL,
        header=None,
        names=column_names,
        na_values="?",
    )
    if len(dataframe) < 200 or dataframe[TARGET_COLUMN].isna().all():
        raise ValueError(
            "Downloaded dataset looks invalid. Check network access to UCI."
        )

    dataframe.to_csv(output_path, index=False)
    print(f"Saved {len(dataframe)} rows to {output_path.resolve()}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCI Heart Disease data")
    parser.add_argument(
        "--output",
        default=DEFAULT_DATA_PATH,
        help="Output CSV path",
    )
    args = parser.parse_args()
    download_heart_disease(args.output)


if __name__ == "__main__":
    main()
