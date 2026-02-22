"""
Visualization utilities for model evaluation and data exploration.

All plotting functions return a ``matplotlib.figure.Figure`` so that callers
can display them in Streamlit (``st.pyplot(fig)``) or save to disk.
"""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.config import FEATURE_ORDER

# Consistent style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ---------------------------------------------------------------------------
# Actual vs Predicted
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Prices",
) -> plt.Figure:
    """Scatter plot with a perfect-fit diagonal reference line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
    lims = [
        min(min(y_true), min(y_pred)),
        max(max(y_true), max(y_pred)),
    ]
    ax.plot(lims, lims, "--", color="red", linewidth=1.5, label="Perfect Fit")
    ax.set_xlabel("Actual Price (Lakhs)")
    ax.set_ylabel("Predicted Price (Lakhs)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Residual Analysis
# ---------------------------------------------------------------------------

def plot_residuals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
) -> plt.Figure:
    """Residuals (y_true − y_pred) vs predicted values."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.5)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Price")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{title} — Residuals vs Predicted")

    # Histogram of residuals
    axes[1].hist(residuals, bins=25, edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{title} — Distribution")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model: Any,
    feature_names: List[str] | None = None,
    title: str = "Feature Importance",
) -> plt.Figure | None:
    """Bar chart of feature importances (tree-based models) or coefficients."""
    feature_names = feature_names or FEATURE_ORDER

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return None  # model does not expose importances

    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[idx], importances[idx], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    all_results: Dict[str, Dict],
    metric: str = "r2",
) -> plt.Figure:
    """Grouped bar chart comparing models on train and test for *metric*."""
    models = list(all_results.keys())
    train_vals = [all_results[m]["train"][metric] for m in models]
    test_vals = [all_results[m]["test"][metric] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, train_vals, width, label="Train", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, test_vals, width, label="Test", color="#dd8452")

    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Comparison — {metric.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()

    # Annotate bar values
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Annotated heatmap of feature correlations."""
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Target Distribution
# ---------------------------------------------------------------------------

def plot_target_distribution(y: pd.Series, title: str = "Selling Price Distribution") -> plt.Figure:
    """Histogram + KDE of the target variable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(y, kde=True, bins=30, ax=ax, color="steelblue")
    ax.set_xlabel("Selling Price (Lakhs)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
