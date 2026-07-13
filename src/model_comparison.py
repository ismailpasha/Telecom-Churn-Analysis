"""
model_comparison.py

Train and compare multiple machine learning models.

Author: Mohammed Ismail Pasha
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from .config import BEST_MODEL_NAME, RANDOM_STATE, REPORT_DIR
    from .data_preprocessing import build_preprocessor
    from .model_utils import save_model
except ImportError:
    from config import BEST_MODEL_NAME, RANDOM_STATE, REPORT_DIR
    from data_preprocessing import build_preprocessor
    from model_utils import save_model


class ModelComparison:

    def __init__(self):

        self.models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE
                )

        }

        self.results = []

    # --------------------------------------------------

    def evaluate_model(
        self,
        model,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(y_test, predictions, pos_label=1, zero_division=0)

        recall = recall_score(y_test, predictions, pos_label=1, zero_division=0)

        f1 = f1_score(y_test, predictions, pos_label=1, zero_division=0)

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        return {

            "Model": pipeline.named_steps["model"].__class__.__name__,

            "Accuracy": round(accuracy, 4),

            "Precision": round(precision, 4),

            "Recall": round(recall, 4),

            "F1 Score": round(f1, 4),

            "ROC AUC": round(roc_auc, 4),

            "Estimator": pipeline

        }

    # --------------------------------------------------

    def compare_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        print("\nTraining Models...\n")

        self.results = []

        for model_name, model in self.models.items():

            print(f"Training {model_name}")

            result = self.evaluate_model(

                model,

                X_train,

                X_test,

                y_train,

                y_test

            )

            result["Display Name"] = model_name

            self.results.append(result)

        results_df = pd.DataFrame(self.results)

        results_df = results_df.sort_values(

            by="F1 Score",

            ascending=False

        )

        return results_df

    # --------------------------------------------------

    def save_results(
        self,
        results_df
    ):

        filepath = REPORT_DIR / "model_comparison.csv"
        legacy_filepath = REPORT_DIR.parent / "model_comparison.csv"

        export_df = results_df.drop(
            columns=["Estimator"]
        )

        export_df.to_csv(
            filepath,
            index=False
        )

        export_df.to_csv(
            legacy_filepath,
            index=False
        )

        print(f"\nResults saved to:\n{filepath}")

    # --------------------------------------------------

    def save_best_model(
        self,
        results_df
    ):

        best = results_df.iloc[0]

        model = best["Estimator"]

        filename = "best_model.pkl"

        save_model(
            model,
            filename
        )

        print(

            f"\nBest Model: {best['Display Name']}"

        )

        print(

            f"F1 Score : {best['F1 Score']}"

        )

    # --------------------------------------------------

    @staticmethod
    def print_results(
        results_df
    ):

        print("\n")

        print("=" * 75)

        print(results_df.drop(
            columns=["Estimator"]
        ))

        print("=" * 75)