"""
business_insights.py

Business Insights Generator

Author: Mohammed Ismail Pasha

Generates executive-level business insights
from churn prediction results.
"""

import pandas as pd

try:
    from .customer_segmentation import CustomerSegmentation
    from .revenue_at_risk import RevenueAtRisk
except ImportError:
    from customer_segmentation import CustomerSegmentation
    from revenue_at_risk import RevenueAtRisk


class BusinessInsights:

    def __init__(self):

        self.segmenter = CustomerSegmentation()

        self.revenue = RevenueAtRisk()

    # ---------------------------------------------------------

    def generate_dashboard(self, dataframe):

        customers = self.segmenter.generate_segments(dataframe)

        customers = self.revenue.calculate_revenue_risk(customers)

        return customers

    # ---------------------------------------------------------

    def executive_summary(self, report):

        total_customers = len(report)

        predicted_churn = len(

            report[
                report["Prediction"] == 1
            ]

        )

        churn_rate = (

            predicted_churn /

            total_customers

        ) * 100

        revenue_risk = report[
            "Revenue At Risk"
        ].sum()

        avg_probability = report[
            "Churn Probability"
        ].mean()

        print("\n")

        print("=" * 70)

        print("EXECUTIVE BUSINESS SUMMARY")

        print("=" * 70)

        print()

        print(f"Customers Analysed : {total_customers}")

        print(f"Predicted Churn : {predicted_churn}")

        print(f"Predicted Churn Rate : {churn_rate:.2f}%")

        print(f"Average Churn Probability : {avg_probability:.2f}%")

        print(f"Monthly Revenue At Risk : ${revenue_risk:,.2f}")

        print()

        print("=" * 70)

    # ---------------------------------------------------------

    def segment_analysis(self, report):

        print("\n")

        print("=" * 70)

        print("CUSTOMER SEGMENT ANALYSIS")

        print("=" * 70)

        print()

        summary = (

            report

            .groupby("Customer Segment")

            .size()

            .reset_index(name="Customers")

        )

        print(summary)

        print()

        print("=" * 70)

    # ---------------------------------------------------------

    def contract_analysis(self, report):

        if "Contract" not in report.columns:

            return

        print("\n")

        print("=" * 70)

        print("HIGH-RISK CONTRACT TYPES")

        print("=" * 70)

        print()

        contract = (

            report

            .groupby("Contract")

            ["Churn Probability"]

            .mean()

            .sort_values(

                ascending=False

            )

        )

        print(contract)

        print()

    # ---------------------------------------------------------

    def payment_analysis(self, report):

        if "PaymentMethod" not in report.columns:

            return

        print("\n")

        print("=" * 70)

        print("PAYMENT METHOD ANALYSIS")

        print("=" * 70)

        print()

        payment = (

            report

            .groupby("PaymentMethod")

            ["Churn Probability"]

            .mean()

            .sort_values(

                ascending=False

            )

        )

        print(payment)

        print()

    # ---------------------------------------------------------

    def top_customers(self, report, top_n=20):

        print("\n")

        print("=" * 70)

        print("TOP CUSTOMERS TO RETAIN")

        print("=" * 70)

        print()

        top = (

            report

            .sort_values(

                by="Revenue At Risk",

                ascending=False

            )

            .head(top_n)

        )

        columns = [

            "customerID",

            "MonthlyCharges",

            "Churn Probability",

            "Revenue At Risk",

            "Customer Segment"

        ]

        available = [

            c for c in columns

            if c in top.columns

        ]

        print(top[available])

    # ---------------------------------------------------------

    def recommendations(self):

        print("\n")

        print("=" * 70)

        print("BUSINESS RECOMMENDATIONS")

        print("=" * 70)

        print()

        recommendations = [

            "Convert month-to-month customers to annual contracts.",

            "Contact Critical and High-risk customers immediately.",

            "Offer discounts to customers with high monthly charges.",

            "Promote Tech Support and Online Security services.",

            "Launch personalized retention campaigns.",

            "Monitor new customers during their first year."

        ]

        for i, rec in enumerate(

            recommendations,

            start=1

        ):

            print(f"{i}. {rec}")

        print()

        print("=" * 70)

    # ---------------------------------------------------------

    def generate_report(self, dataframe):

        report = self.generate_dashboard(dataframe)

        self.executive_summary(report)

        self.segment_analysis(report)

        self.contract_analysis(report)

        self.payment_analysis(report)

        self.top_customers(report)

        self.recommendations()

        return report