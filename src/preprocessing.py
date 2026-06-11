"""Reusable preprocessing utilities and sklearn pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.constants import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    MISSING_THRESHOLD,
    TARGET_COLUMN,
)


class DropDuplicatesTransformer(BaseEstimator, TransformerMixin):
    """Remove duplicate rows using only training data statistics."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.drop_duplicates().reset_index(drop=True)
        return X


class DropHighMissingColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drop columns whose missing rate exceeds the configured threshold."""

    def __init__(self, threshold: float = MISSING_THRESHOLD):
        self.threshold = threshold
        self.columns_to_keep_: list[str] = []

    def fit(self, X, y=None):
        frame = self._to_frame(X)
        missing_rates = frame.isna().mean()
        self.columns_to_keep_ = [
            column
            for column in frame.columns
            if missing_rates[column] <= self.threshold
        ]
        return self

    def transform(self, X):
        frame = self._to_frame(X)
        return frame[self.columns_to_keep_]

    @staticmethod
    def _to_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=FEATURE_COLUMNS)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip continuous features to train-derived IQR bounds."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_: dict[str, float] = {}
        self.upper_bounds_: dict[str, float] = {}

    def fit(self, X, y=None):
        frame = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        for column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[column] = q1 - self.factor * iqr
            self.upper_bounds_[column] = q3 + self.factor * iqr
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, columns=FEATURE_COLUMNS).copy()
        for column in frame.columns:
            lower = self.lower_bounds_[column]
            upper = self.upper_bounds_[column]
            frame[column] = frame[column].clip(lower=lower, upper=upper)
        return frame


def load_raw_dataframe(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the dataset and normalize missing tokens to np.nan."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python data/download_data.py"
        )

    dataframe = pd.read_csv(path)
    dataframe.replace("?", np.nan, inplace=True)
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe


def binarize_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert multi-class UCI target into binary labels (0=healthy, 1=disease)."""
    frame = dataframe.copy()
    frame[TARGET_COLUMN] = (frame[TARGET_COLUMN] > 0).astype(int)
    return frame


def split_features_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and binary target y."""
    features = dataframe[FEATURE_COLUMNS].copy()
    target = dataframe[TARGET_COLUMN].astype(int)
    return features, target


def build_preprocessing_pipeline(
  numeric_columns: Iterable[str] | None = None,
  categorical_columns: Iterable[str] | None = None,
) -> Pipeline:
    """
    Build a leak-safe preprocessing pipeline.

    All transformers are sklearn-compatible and must be fit only on training data.
    """
    numeric_columns = list(numeric_columns or FEATURE_COLUMNS)
    categorical_columns = list(categorical_columns or [])

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", IQRClipper()),
    ]
    numeric_pipeline = Pipeline(numeric_steps)

    transformers = [("numeric", numeric_pipeline, numeric_columns)]
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="most_frequent"))]
        )
        transformers.append(
            ("categorical", categorical_pipeline, list(categorical_columns))
        )

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("drop_duplicates", DropDuplicatesTransformer()),
            ("drop_high_missing", DropHighMissingColumnsTransformer()),
            ("column_transformer", column_transformer),
        ]
    )


def get_feature_names_after_preprocessing() -> list[str]:
    """Return feature names preserved by the default preprocessing pipeline."""
    return FEATURE_COLUMNS.copy()
