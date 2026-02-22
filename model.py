"""
Model training, persistence, and loading utilities.

Supports multiple regression algorithms.  The ``train_all_models`` function
trains every model defined in ``config.MODEL_CONFIGS`` and returns them
along with their predictions so that downstream evaluation can pick the best.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from config.config import (
    DEFAULT_MODEL_NAME,
    MODEL_ARTIFACT_PATH,
    MODEL_CONFIGS,
    METRICS_ARTIFACT_PATH,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

_MODEL_CLASSES: Dict[str, Any] = {
    "LinearRegression": LinearRegression,
    "Lasso": Lasso,
    "Ridge": Ridge,
    "RandomForest": RandomForestRegressor,
    "GradientBoosting": GradientBoostingRegressor,
}


def _get_model(name: str) -> Any:
    """Instantiate a model by name with its configured hyperparameters."""
    if name not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_CLASSES.keys())}"
        )
    params = MODEL_CONFIGS.get(name, {})
    model = _MODEL_CLASSES[name](**params)
    logger.info("Instantiated %s with params %s", name, params)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """Train a single model and return the fitted estimator."""
    model = _get_model(name)
    model.fit(X_train, y_train)
    logger.info("Trained %s on %d samples", name, len(X_train))
    return model


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Any]:
    """Train every model in ``MODEL_CONFIGS`` and return a dict of fitted estimators."""
    models: Dict[str, Any] = {}
    for name in MODEL_CONFIGS:
        models[name] = train_model(name, X_train, y_train)
    return models


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    path: Path | str | None = None,
    metadata: dict | None = None,
) -> Path:
    """Persist a trained model (and optional metadata) to disk.

    Parameters
    ----------
    model : estimator
        Fitted scikit-learn model.
    path : Path or str, optional
        Destination file.  Defaults to ``MODEL_ARTIFACT_PATH``.
    metadata : dict, optional
        Additional info (metrics, params) saved alongside the model.

    Returns
    -------
    Path
        The path to the saved artifact.
    """
    path = Path(path) if path else MODEL_ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {"model": model, "metadata": metadata or {}}
    joblib.dump(artifact, path)
    logger.info("Model saved to %s", path)
    return path


def load_model(path: Path | str | None = None) -> Tuple[Any, dict]:
    """Load a model and its metadata from disk.

    Returns
    -------
    model : estimator
        The fitted scikit-learn estimator.
    metadata : dict
        Any metadata saved alongside the model.
    """
    path = Path(path) if path else MODEL_ARTIFACT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {path}")

    artifact = joblib.load(path)

    # Support loading legacy plain-model pickles (e.g. car_model.pkl)
    if not isinstance(artifact, dict):
        logger.warning("Loaded a legacy model without metadata wrapper.")
        return artifact, {}

    return artifact["model"], artifact.get("metadata", {})


def save_metrics(metrics: dict, path: Path | str | None = None) -> Path:
    """Save evaluation metrics as a JSON file."""
    path = Path(path) if path else METRICS_ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", path)
    return path
