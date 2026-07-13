"""
train_models.py

Train and compare churn prediction models, then persist
the best pipeline and the standard project reports.
"""

from __future__ import annotations

import time

from sklearn.metrics import classification_report

try:
    from .config import PROCESSED_DATA_PATH, REPORT_DIR, TARGET_COLUMN
    from .feature_engineering import FeatureEngineer
    from .logger import get_logger
    from .model_comparison import ModelComparison
except ImportError:
    from config import PROCESSED_DATA_PATH, REPORT_DIR, TARGET_COLUMN
    from feature_engineering import FeatureEngineer
    from logger import get_logger
    from model_comparison import ModelComparison


logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.comparison = ModelComparison()
        self.data = None
        self.results = None
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_dataset(self):
        source = PROCESSED_DATA_PATH if PROCESSED_DATA_PATH.exists() else None
        self.data = self.feature_engineer.load_dataset(source)
        logger.info("Loaded %s", source or "raw dataset")

    def validate_dataset(self):
        if self.data is None:
            raise ValueError("Dataset has not been loaded.")

        if TARGET_COLUMN not in self.data.columns:
            raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    def preprocess(self):
        self.data = self.feature_engineer.prepare_dataframe(self.data)

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.feature_engineer.prepare_dataset(
            self.data
        )

    def save_processed_dataset(self):
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info("Processed dataset saved to %s", PROCESSED_DATA_PATH)

    def train_and_compare(self):
        self.results = self.comparison.compare_models(
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        )
        self.best_model = self.results.iloc[0]["Estimator"]

    def save_reports(self):
        self.comparison.print_results(self.results)
        self.comparison.save_results(self.results)
        self.comparison.save_best_model(self.results)

    def save_test_predictions(self):
        predictions = self.best_model.predict(self.X_test)
        probabilities = self.best_model.predict_proba(self.X_test)[:, 1]

        output = self.X_test.copy().reset_index(drop=True)
        output["Actual"] = self.y_test.reset_index(drop=True)
        output["Prediction"] = predictions
        output["Churn Probability"] = (probabilities * 100).round(2)

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        predictions_path = REPORT_DIR.parent / "test_predictions.csv"
        output.to_csv(predictions_path, index=False)
        logger.info("Test predictions saved to %s", predictions_path)

        report_path = REPORT_DIR / "classification_report.txt"
        legacy_report_path = REPORT_DIR.parent / "classification_report.txt"
        report_text = classification_report(self.y_test, predictions, digits=4)
        report_path.write_text(report_text, encoding="utf-8")
        legacy_report_path.write_text(report_text, encoding="utf-8")
        logger.info("Classification report saved to %s", report_path)

    def run(self):
        start = time.time()
        try:
            self.load_dataset()
            self.validate_dataset()
            self.preprocess()
            self.save_processed_dataset()
            self.split_data()
            self.train_and_compare()
            self.save_reports()
            self.save_test_predictions()
            logger.info("Training pipeline completed successfully")
        except Exception:
            logger.exception("Training failed")
            raise
        finally:
            logger.info("Elapsed %.2f sec", time.time() - start)


if __name__ == "__main__":
    ModelTrainer().run()
