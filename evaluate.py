"""
Model evaluation and metrics utilities.

Computes a comprehensive set of regression metrics and supports
multi-model comparison to select the best estimator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute a full suite of regression metrics.

    Returns
    -------
    dict
        Keys: ``r2``, ``mae``, ``mse``, ``rmse``, ``mape``.
    """
    metrics = {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred)), 4),
    }
    return metrics


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a single model on train and test sets.

    Returns
    -------
    dict
        ``{"train": {...metrics}, "test": {...metrics}}``.
    """
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    results = {
        "train": compute_metrics(y_train, train_preds),
        "test": compute_metrics(y_test, test_preds),
    }
    logger.info(
        "Evaluation — Train R²: %.4f | Test R²: %.4f",
        results["train"]["r2"],
        results["test"]["r2"],
    )
    return results


def compare_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selection_metric: str = "r2",
) -> tuple[str, Any, Dict[str, Dict]]:
    """Evaluate all models and select the best based on *selection_metric*.

    Parameters
    ----------
    models : dict
        Mapping of model names to fitted estimators.
    selection_metric : str
        Metric to maximise for model selection (default ``"r2"``).

    Returns
    -------
    best_name : str
    best_model : estimator
    all_results : dict
        ``{model_name: {"train": {...}, "test": {...}}}``.
    """
    all_results: Dict[str, Dict] = {}
    for name, model in models.items():
        all_results[name] = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Select best on test-set metric (higher is better for r2; if using error
    # metrics the comparison should be inverted — keeping it simple for r2)
    best_name = max(
        all_results,
        key=lambda n: all_results[n]["test"][selection_metric],
    )
    best_model = models[best_name]
    logger.info("Best model: %s (test %s = %.4f)",
                best_name, selection_metric,
                all_results[best_name]["test"][selection_metric])
    return best_name, best_model, all_results


def results_to_dataframe(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """Flatten the nested results dict into a tidy DataFrame for display."""
    rows = []
    for name, splits in all_results.items():
        for split, metrics in splits.items():
            rows.append({"Model": name, "Split": split, **metrics})
    return pd.DataFrame(rows)
