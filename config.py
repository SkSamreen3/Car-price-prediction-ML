"""
Configuration module for Car Price Prediction project.

Centralizes all constants, paths, hyperparameters, and encoding maps
to ensure consistency across training and inference pipelines.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data_Files"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DATA_PATH = DATA_DIR / "car data.csv"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data Schema
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Selling_Price"
DROP_COLUMNS = ["Car_Name"]  # columns removed before training
FEATURE_ORDER = [
    "Year",
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
]

# ---------------------------------------------------------------------------
# Categorical Encoding Maps  (label-encoding used in both train & inference)
# ---------------------------------------------------------------------------
ENCODING_MAPS: dict[str, dict[str, int]] = {
    "Fuel_Type": {"Petrol": 0, "Diesel": 1, "CNG": 2},
    "Seller_Type": {"Dealer": 0, "Individual": 1},
    "Transmission": {"Manual": 0, "Automatic": 1},
}

# Reverse maps (for UI display / decoding)
DECODING_MAPS: dict[str, dict[int, str]] = {
    col: {v: k for k, v in mapping.items()}
    for col, mapping in ENCODING_MAPS.items()
}

# ---------------------------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------------------------
TEST_SIZE = 0.1
RANDOM_STATE = 42

MODEL_CONFIGS = {
    "LinearRegression": {},
    "Lasso": {"alpha": 1.0},
    "Ridge": {"alpha": 1.0},
    "RandomForest": {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": RANDOM_STATE,
    },
    "GradientBoosting": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 4,
        "random_state": RANDOM_STATE,
    },
}

# Default model used for the Streamlit app
DEFAULT_MODEL_NAME = "RandomForest"
MODEL_ARTIFACT_PATH = MODELS_DIR / "best_model.joblib"
METRICS_ARTIFACT_PATH = MODELS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Streamlit UI Defaults
# ---------------------------------------------------------------------------
YEAR_RANGE = (1990, 2025)
PRESENT_PRICE_RANGE = (0.0, 100.0)
KMS_DRIVEN_RANGE = (0, 500_000)
OWNER_OPTIONS = [0, 1, 2, 3]
