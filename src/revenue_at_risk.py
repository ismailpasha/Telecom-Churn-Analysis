"""
revenue_at_risk.py

Estimate monthly revenue at risk using churn probabilities.

Author: Mohammed Ismail Pasha
"""

from pathlib import Path
import pandas as pd

try:
    from .predict import ChurnPredictor
    from .config import REPORT_DIR
except ImportError:
    from predict import ChurnPredictor
    from config import REPORT_DIR


class RevenueAtRisk:

    def __init__(self):

        self.predictor = ChurnPredictor()

    # -----------------------------------------------------

    @staticmethod
    def calculate_revenue_risk(df):

        df = df.copy()

        df["Revenue At Risk"] = (
            df["MonthlyCharges"]
            * (df["Churn Probability"] / 100)
        ).round(2)

        return df

    # -----------------------------------------------------

    @staticmethod
    def retention_priority(probability):

        if probability >= 80:
            return "Immediate"

        elif probability >= 60:
            return "High"

        elif probability >= 40:
            return "Medium"

        return "Low"

    # -----------------------------------------------------

    def generate_report(self, dataframe):

        results = self.predictor.generate_predictions(dataframe)

        results = self.calculate_revenue_risk(results)

        results["Retention Priority"] = results[
            "Churn Probability"
        ].apply(self.retention_priority)

        return results

    # -----------------------------------------------------

    def top_customers(
        self,
        report,
        top_n=20
    ):

        return report.sort_values(

            by="Revenue At Risk",

            ascending=False

        ).head(top_n)

    # -----------------------------------------------------

    def business_summary(self, report):

        total_revenue = report[
            "Revenue At Risk"
        ].sum()

        high_risk = len(

            report[
                report["Retention Priority"] == "Immediate"
            ]

        )

        print("\n" + "=" * 60)

        print("MONTHLY REVENUE AT RISK")

        print("=" * 60)

        print()

        print(
            f"Customers Analysed : {len(report)}"
        )

        print()

        print(
            f"Immediate Retention Required : {high_risk}"
        )

        print()

        print(
            f"Estimated Monthly Revenue At Risk : "
            f"${total_revenue:,.2f}"
        )

        print("\n" + "=" * 60)

    # -----------------------------------------------------

    def save_report(
        self,
        report
    ):

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        filepath = REPORT_DIR / "revenue_at_risk.csv"

        report.to_csv(
            filepath,
            index=False
        )

        print(f"\nSaved report:\n{filepath}")


if __name__ == "__main__":

    print(
        "RevenueAtRisk module loaded.\n"
        "Use generate_report() with preprocessed data."
    )