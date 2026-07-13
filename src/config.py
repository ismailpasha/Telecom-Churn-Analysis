"""
config.py

Central configuration module for the Telecom Churn Prediction project.

Author: Mohammed Ismail Pasha

This module centralizes:
- Project metadata
- Directory paths
- Dataset paths
- Model settings
- Training settings
- Output file locations
- Environment variable overrides

Every module in the project should import configuration
from this file instead of hardcoding paths.
"""

from dataclasses import dataclass
from pathlib import Path
import os

# ==============================================================================
# PROJECT INFORMATION
# ==============================================================================

PROJECT_NAME = "Telecom Churn Prediction"

PROJECT_DESCRIPTION = (
    "Machine Learning pipeline for predicting telecom customer churn "
    "using classification models and business analytics."
)

AUTHOR = "Mohammed Ismail Pasha"

VERSION = "1.0.0"

# ==============================================================================
# PROJECT ROOT
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==============================================================================
# DIRECTORIES
# ==============================================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"

REPORT_DIR = OUTPUT_DIR / "reports"

FIGURE_DIR = OUTPUT_DIR / "figures"

PREDICTION_DIR = OUTPUT_DIR / "predictions"

LOG_DIR = PROJECT_ROOT / "logs"

DOCS_DIR = PROJECT_ROOT / "docs"

TEST_DIR = PROJECT_ROOT / "tests"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

# ==============================================================================
# DATASET
# ==============================================================================

DATASET_NAME = os.getenv(
    "DATASET_NAME",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

RAW_DATA_PATH = RAW_DATA_DIR / DATASET_NAME

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR /
    "telco_churn_processed.csv"
)

# ==============================================================================
# MACHINE LEARNING SETTINGS
# ==============================================================================

TARGET_COLUMN = os.getenv("TARGET_COLUMN", "Churn")

TEST_SIZE = float(
    os.getenv("TEST_SIZE", 0.20)
)

RANDOM_STATE = int(
    os.getenv("RANDOM_STATE", 42)
)

CV_FOLDS = int(
    os.getenv("CV_FOLDS", 5)
)

SCORING_METRIC = os.getenv(
    "SCORING_METRIC",
    "f1"
)

# ==============================================================================
# MODEL FILES
# ==============================================================================

PIPELINE_NAME = "telecom_churn_pipeline.pkl"

PIPELINE_PATH = MODEL_DIR / PIPELINE_NAME

BEST_MODEL_NAME = "best_model.pkl"

BEST_MODEL_PATH = MODEL_DIR / BEST_MODEL_NAME

# ==============================================================================
# REPORT FILES
# ==============================================================================

MODEL_METRICS_REPORT = (
    REPORT_DIR /
    "model_metrics.csv"
)

BEST_PARAMETERS_REPORT = (
    REPORT_DIR /
    "best_parameters.csv"
)

CROSS_VALIDATION_REPORT = (
    REPORT_DIR /
    "cross_validation.csv"
)

CLASSIFICATION_REPORT = (
    REPORT_DIR /
    "classification_report.txt"
)

FEATURE_IMPORTANCE_REPORT = (
    REPORT_DIR /
    "feature_importance.csv"
)

REVENUE_REPORT = (
    REPORT_DIR /
    "revenue_at_risk.csv"
)

CUSTOMER_SEGMENT_REPORT = (
    REPORT_DIR /
    "customer_segments.csv"
)

# ==============================================================================
# FIGURES
# ==============================================================================

CONFUSION_MATRIX_FIG = (
    FIGURE_DIR /
    "confusion_matrix.png"
)

ROC_CURVE_FIG = (
    FIGURE_DIR /
    "roc_curve.png"
)

FEATURE_IMPORTANCE_FIG = (
    FIGURE_DIR /
    "feature_importance.png"
)

CORRELATION_HEATMAP_FIG = (
    FIGURE_DIR /
    "correlation_heatmap.png"
)

# ==============================================================================
# PREDICTIONS
# ==============================================================================

PREDICTIONS_FILE = (
    PREDICTION_DIR /
    "customer_predictions.csv"
)

# ==============================================================================
# LOGGING
# ==============================================================================

LOG_FILE = LOG_DIR / "telecom_churn.log"

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO"
)

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

MODELS = [

    "Logistic Regression",

    "Decision Tree",

    "Random Forest"

]

# ==============================================================================
# OPTIONAL DATACLASS
# ==============================================================================

@dataclass(frozen=True)
class TrainingConfig:
    """
    Centralized training configuration.
    """

    target_column: str = TARGET_COLUMN

    test_size: float = TEST_SIZE

    random_state: int = RANDOM_STATE

    cv_folds: int = CV_FOLDS

    scoring_metric: str = SCORING_METRIC

# ==============================================================================
# CREATE DIRECTORIES
# ==============================================================================

DIRECTORIES = [

    DATA_DIR,

    RAW_DATA_DIR,

    PROCESSED_DATA_DIR,

    OUTPUT_DIR,

    MODEL_DIR,

    REPORT_DIR,

    FIGURE_DIR,

    PREDICTION_DIR,

    LOG_DIR,

    DOCS_DIR,

    TEST_DIR,

    NOTEBOOK_DIR

]

for directory in DIRECTORIES:

    directory.mkdir(

        parents=True,

        exist_ok=True

    )

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print(PROJECT_NAME)
    print("=" * 70)
    print(f"Version           : {VERSION}")
    print(f"Author            : {AUTHOR}")
    print(f"Project Root      : {PROJECT_ROOT}")
    print(f"Raw Dataset       : {RAW_DATA_PATH}")
    print(f"Processed Dataset : {PROCESSED_DATA_PATH}")
    print(f"Model Directory   : {MODEL_DIR}")
    print(f"Reports Directory : {REPORT_DIR}")
    print(f"Figures Directory : {FIGURE_DIR}")
    print(f"Logs Directory    : {LOG_DIR}")
    print("=" * 70)