"""
feature_importance.py

Generate Feature Importance Report
for Telecom Churn Prediction.

Author: Mohammed Ismail Pasha
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .config import FIGURE_DIR, REPORT_DIR
    from .model_utils import load_model
except ImportError:
    from config import FIGURE_DIR, REPORT_DIR
    from model_utils import load_model


class FeatureImportance:

    def __init__(self, model_name="best_model.pkl"):

        self.model = load_model(model_name)

    def _get_estimator(self):
        if hasattr(self.model, "named_steps"):
            return self.model.named_steps["model"]
        return self.model

    def _get_feature_names(self):
        if hasattr(self.model, "named_steps") and "preprocessor" in self.model.named_steps:
            return self.model.named_steps["preprocessor"].get_feature_names_out()
        return None

    # -----------------------------------------------------

    def get_feature_importance(self, feature_names):

        estimator = self._get_estimator()

        resolved_feature_names = self._get_feature_names()

        if resolved_feature_names is None:
            resolved_feature_names = feature_names

        if resolved_feature_names is not None and not isinstance(resolved_feature_names, list):
            resolved_feature_names = list(resolved_feature_names)

        if resolved_feature_names is None:
            raise ValueError(
                "Feature names could not be resolved for this model."
            )

        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            values = abs(estimator.coef_).ravel()
        else:
            raise ValueError(
                "This model does not support feature importance."
            )

        importance = pd.DataFrame({

            "Feature": resolved_feature_names,

            "Importance": values

        })

        importance = importance.sort_values(

            by="Importance",

            ascending=False

        ).reset_index(drop=True)

        return importance

    # -----------------------------------------------------

    def save_csv(self, importance_df):

        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        filepath = REPORT_DIR / "feature_importance.csv"

        importance_df.to_csv(filepath, index=False)

        print(f"✓ Feature importance saved to:\n{filepath}")

    # -----------------------------------------------------

    def plot(self, importance_df, top_n=15):

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 7))

        top_features = importance_df.head(top_n)

        plt.barh(

            top_features["Feature"][::-1],

            top_features["Importance"][::-1]

        )

        plt.xlabel("Importance")

        plt.ylabel("Features")

        plt.title("Top Feature Importance")

        plt.tight_layout()

        filepath = FIGURE_DIR / "feature_importance.png"

        plt.savefig(filepath, dpi=300)

        plt.close()

        print(f"✓ Figure saved to:\n{filepath}")

    # -----------------------------------------------------

    def print_summary(self, importance_df, top_n=10):

        print("\n")

        print("=" * 60)

        print("TOP FEATURES CONTRIBUTING TO CUSTOMER CHURN")

        print("=" * 60)

        print()

        print(importance_df.head(top_n))

        print()

        print("=" * 60)

    # -----------------------------------------------------

    def generate(self, feature_names):

        importance = self.get_feature_importance(feature_names)

        self.print_summary(importance)

        self.save_csv(importance)

        self.plot(importance)

        return importance