"""
customer_segmentation.py

Customer Risk Segmentation Module

Author: Mohammed Ismail Pasha

Segments customers into business-friendly
risk groups based on churn probability.
"""

import pandas as pd

from pathlib import Path

try:
    from .predict import ChurnPredictor
    from .config import REPORT_DIR
except ImportError:
    from predict import ChurnPredictor
    from config import REPORT_DIR


class CustomerSegmentation:

    def __init__(self):

        self.predictor = ChurnPredictor()

    # --------------------------------------------------

    @staticmethod
    def segment(probability):

        """
        Segment customer according to
        churn probability.
        """

        if probability >= 90:

            return "Critical"

        elif probability >= 75:

            return "High"

        elif probability >= 50:

            return "Medium"

        elif probability >= 25:

            return "Low"

        return "Safe"

    # --------------------------------------------------

    @staticmethod
    def recommended_action(segment):

        recommendations = {

            "Critical":
                "Immediate phone call and special retention offer",

            "High":
                "Offer annual contract and loyalty discount",

            "Medium":
                "Email campaign with personalized offers",

            "Low":
                "Monitor customer engagement",

            "Safe":
                "No immediate action required"

        }

        return recommendations[segment]

    # --------------------------------------------------

    def generate_segments(self, dataframe):

        predictions = self.predictor.generate_predictions(
            dataframe
        )

        predictions["Customer Segment"] = (

            predictions["Churn Probability"]

            .apply(self.segment)

        )

        predictions["Recommended Action"] = (

            predictions["Customer Segment"]

            .apply(self.recommended_action)

        )

        return predictions

    # --------------------------------------------------

    def segment_summary(self, dataframe):

        summary = (

            dataframe

            .groupby("Customer Segment")

            .size()

            .reset_index(name="Customers")

            .sort_values(

                by="Customers",

                ascending=False

            )

        )

        return summary

    # --------------------------------------------------

    def save_segments(self, dataframe):

        REPORT_DIR.mkdir(

            parents=True,

            exist_ok=True

        )

        filepath = (

            REPORT_DIR /

            "customer_segments.csv"

        )

        dataframe.to_csv(

            filepath,

            index=False

        )

        print(

            f"\nCustomer Segments Saved:\n{filepath}"

        )

    # --------------------------------------------------

    def print_summary(self, summary):

        print("\n")

        print("=" * 60)

        print("CUSTOMER SEGMENTATION REPORT")

        print("=" * 60)

        print()

        print(summary)

        print()

        print("=" * 60)

    # --------------------------------------------------

    def top_priority_customers(

        self,

        dataframe,

        top_n=20

    ):

        priority = (

            dataframe

            .sort_values(

                by="Churn Probability",

                ascending=False

            )

            .head(top_n)

        )

        return priority


if __name__ == "__main__":

    print(

        "Customer Segmentation Module Loaded."

    )