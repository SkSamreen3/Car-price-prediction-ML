"""
Data loading, validation, and preprocessing utilities.

Responsibilities:
    - Load raw CSV data with validation
    - Encode categorical features using project-wide encoding maps
    - Split data into features / target and train / test sets
    - Build a single-row DataFrame for inference from user inputs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    DEFAULT_DATA_PATH,
    DROP_COLUMNS,
    ENCODING_MAPS,
    FEATURE_ORDER,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the car-price dataset from *path* (defaults to ``DEFAULT_DATA_PATH``).

    Parameters
    ----------
    path : Path or str, optional
        Path to the CSV file.  Falls back to the project default.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame exactly as stored on disk.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not point to a file.
    """
    path = Path(path) if path else DEFAULT_DATA_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    logger.info("Loaded dataset: %s rows, %s columns from %s", *df.shape, path.name)
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply label encoding to categorical columns **in-place copy**.

    Uses the shared ``ENCODING_MAPS`` from config so that training and
    inference always agree on the mapping.
    """
    df = df.copy()
    for col, mapping in ENCODING_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(int)
    logger.info("Encoded categorical columns: %s", list(ENCODING_MAPS.keys()))
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the raw DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing or data contains unexpected nulls.
    """
    required = set(FEATURE_ORDER + [TARGET_COLUMN] + DROP_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    null_counts = df[FEATURE_ORDER + [TARGET_COLUMN]].isnull().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        logger.warning("Null values detected:\n%s", nulls)


def prepare_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop non-feature columns and separate X / y.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded DataFrame (after ``encode_categoricals``).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns in ``FEATURE_ORDER``.
    y : pd.Series
        Target (selling price).
    """
    X = df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN], errors="ignore")
    X = X[FEATURE_ORDER]
    y = df[TARGET_COLUMN]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Wrapper around ``train_test_split`` with project defaults."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Train / test split: %d train, %d test (%.0f%% test)",
        len(X_train),
        len(X_test),
        test_size * 100,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Inference Helper
# ---------------------------------------------------------------------------

def build_inference_input(
    year: int,
    present_price: float,
    kms_driven: int,
    fuel_type: str,
    seller_type: str,
    transmission: str,
    owner: int,
) -> pd.DataFrame:
    """Construct a single-row DataFrame for model prediction.

    String values are automatically encoded using ``ENCODING_MAPS``.
    """
    row = {
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": ENCODING_MAPS["Fuel_Type"][fuel_type],
        "Seller_Type": ENCODING_MAPS["Seller_Type"][seller_type],
        "Transmission": ENCODING_MAPS["Transmission"][transmission],
        "Owner": owner,
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)
