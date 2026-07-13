"""
predict.py

Generate predictions for new customers.

Author: Mohammed Ismail Pasha
"""

from __future__ import annotations

import pandas as pd

try:
    from .config import PREDICTION_DIR
    from .data_preprocessing import clean_data, engineer_features
    from .model_utils import load_model
except ImportError:
    from config import PREDICTION_DIR
    from data_preprocessing import clean_data, engineer_features
    from model_utils import load_model


class ChurnPredictor:

    def __init__(self):

        self.model = load_model("best_model.pkl")

    def _prepare_input(self, dataframe):
        prepared = clean_data(dataframe)
        prepared = engineer_features(prepared)

        if "Churn" in prepared.columns:
            prepared = prepared.drop(columns=["Churn"])

        return prepared

    # ---------------------------------------------------------

    def predict(self, X):

        return self.model.predict(self._prepare_input(X))

    # ---------------------------------------------------------

    def predict_probability(self, X):

        return self.model.predict_proba(self._prepare_input(X))

    # ---------------------------------------------------------

    @staticmethod
    def risk_level(probability):

        if probability >= 0.80:

            return "HIGH"

        elif probability >= 0.50:

            return "MEDIUM"

        return "LOW"

    # ---------------------------------------------------------

    def generate_predictions(self, dataframe):

        prepared = self._prepare_input(dataframe)

        predictions = self.model.predict(prepared)

        probabilities = self.model.predict_proba(prepared)

        results = dataframe.copy()

        if "Churn" in results.columns:
            results = results.drop(columns=["Churn"])

        results["Prediction"] = predictions

        results["Churn Probability"] = (
            probabilities[:, 1] * 100
        ).round(2)

        results["Risk Level"] = results[
            "Churn Probability"
        ].apply(
            lambda x: self.risk_level(x / 100)
        )

        return results

    # ---------------------------------------------------------

    def save_predictions(
        self,
        results,
        filename="customer_predictions.csv"
    ):

        filepath = PREDICTION_DIR / filename

        results.to_csv(filepath, index=False)

        print(f"\nPredictions saved to:\n{filepath}")

    # ---------------------------------------------------------

    def print_summary(self, results):

        print("\n" + "=" * 50)

        print(" TELECOM CHURN PREDICTION REPORT ")

        print("=" * 50)

        print()

        print(
            "Customers Analysed :",
            len(results)
        )

        print()

        print(
            "High Risk Customers :",
            len(results[
                results["Risk Level"] == "HIGH"
            ])
        )

        print()

        print(
            "Medium Risk Customers :",
            len(results[
                results["Risk Level"] == "MEDIUM"
            ])
        )

        print()

        print(
            "Low Risk Customers :",
            len(results[
                results["Risk Level"] == "LOW"
            ])
        )

        print()

        print("=" * 50)


if __name__ == "__main__":

    print(
        "\nUse this class from another script "
        "after preprocessing customer data."
    )