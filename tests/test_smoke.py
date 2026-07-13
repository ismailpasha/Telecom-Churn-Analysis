from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PROCESSED_DATA_PATH, RAW_DATA_PATH, TARGET_COLUMN
from data_preprocessing import load_data
from feature_engineering import FeatureEngineer
from model_comparison import ModelComparison


def test_dataset_loads_from_raw():
    df = load_data(RAW_DATA_PATH)
    assert not df.empty
    assert TARGET_COLUMN in df.columns


def test_feature_engineering_produces_expected_columns():
    engineer = FeatureEngineer()
    df = engineer.load_dataset()
    prepared = engineer.prepare_dataframe(df)
    assert TARGET_COLUMN in prepared.columns
    assert "is_new_customer" in prepared.columns
    assert "avg_charge_per_month" in prepared.columns
    assert "num_services" in prepared.columns


def test_model_comparison_runs_and_saves_best_model():
    engineer = FeatureEngineer()
    df = engineer.load_dataset()
    X_train, X_test, y_train, y_test = engineer.prepare_dataset(df)

    comparison = ModelComparison()
    results = comparison.compare_models(X_train, X_test, y_train, y_test)

    assert not results.empty
    assert "F1 Score" in results.columns
    assert results.iloc[0]["Estimator"] is not None


def test_processed_dataset_path_is_under_data_directory():
    assert PROCESSED_DATA_PATH.parent.name == "processed"
