#!/usr/bin/env python
"""
Training pipeline for Car Price Prediction.

Usage:
    python train.py                  # train all models, save the best
    python train.py --model Lasso    # train only Lasso

This script orchestrates data loading ➜ preprocessing ➜ training ➜
evaluation ➜ model selection ➜ artifact persistence.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so config/src can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DEFAULT_DATA_PATH, FEATURE_ORDER, MODEL_CONFIGS
from src.data_preprocessing import (
    encode_categoricals,
    load_data,
    prepare_features_target,
    split_data,
    validate_dataframe,
)
from src.evaluate import compare_models, evaluate_model, results_to_dataframe
from src.model import save_metrics, save_model, train_all_models, train_model
from src.visualize import (
    plot_actual_vs_predicted,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
    plot_target_distribution,
)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def main(data_path: str | None = None, model_name: str | None = None) -> None:
    """Run the full training pipeline."""

    # 1 — Load & validate
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)
    df = load_data(data_path)
    validate_dataframe(df)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))
    logger.info("Null counts:\n%s", df.isnull().sum().to_string())

    # 2 — Preprocess
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing")
    logger.info("=" * 60)
    df_encoded = encode_categoricals(df)
    X, y = prepare_features_target(df_encoded)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3 — Train
    logger.info("=" * 60)
    logger.info("STEP 3: Training")
    logger.info("=" * 60)
    if model_name:
        model = train_model(model_name, X_train, y_train)
        results = evaluate_model(model, X_train, y_train, X_test, y_test)
        best_name, best_model = model_name, model
        all_results = {model_name: results}
    else:
        models = train_all_models(X_train, y_train)
        best_name, best_model, all_results = compare_models(
            models, X_train, y_train, X_test, y_test
        )

    # 4 — Report
    logger.info("=" * 60)
    logger.info("STEP 4: Evaluation Summary")
    logger.info("=" * 60)
    df_results = results_to_dataframe(all_results)
    logger.info("\n%s", df_results.to_string(index=False))
    logger.info("Best model: %s", best_name)

    # 5 — Save artifacts
    logger.info("=" * 60)
    logger.info("STEP 5: Saving artifacts")
    logger.info("=" * 60)
    metadata = {
        "best_model": best_name,
        "features": FEATURE_ORDER,
        "test_metrics": all_results[best_name]["test"],
        "train_metrics": all_results[best_name]["train"],
    }
    save_model(best_model, metadata=metadata)
    save_metrics(
        {name: res for name, res in all_results.items()},
    )

    # 6 — Save plots (optional, non-blocking)
    logger.info("=" * 60)
    logger.info("STEP 6: Generating visualizations")
    logger.info("=" * 60)
    plots_dir = PROJECT_ROOT / "models" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_test_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)

    fig = plot_actual_vs_predicted(y_test, y_test_pred, f"{best_name} — Test Set")
    fig.savefig(plots_dir / "actual_vs_predicted_test.png", dpi=150)

    fig = plot_residuals(y_test, y_test_pred, f"{best_name} — Test Set")
    fig.savefig(plots_dir / "residuals_test.png", dpi=150)

    fig = plot_feature_importance(best_model, FEATURE_ORDER, f"{best_name} — Feature Importance")
    if fig:
        fig.savefig(plots_dir / "feature_importance.png", dpi=150)

    if len(all_results) > 1:
        fig = plot_model_comparison(all_results, metric="r2")
        fig.savefig(plots_dir / "model_comparison_r2.png", dpi=150)

    fig = plot_correlation_heatmap(df_encoded)
    fig.savefig(plots_dir / "correlation_heatmap.png", dpi=150)

    fig = plot_target_distribution(y)
    fig.savefig(plots_dir / "target_distribution.png", dpi=150)

    plt_close_all()
    logger.info("Plots saved to %s", plots_dir)
    logger.info("✅ Training pipeline complete!")


def plt_close_all():
    """Close all matplotlib figures to free memory."""
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train car price prediction models.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=f"Path to CSV dataset (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Train a single model instead of all. Omit to train & compare all.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(data_path=args.data, model_name=args.model)
