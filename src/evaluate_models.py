from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    comparison_path = OUTPUTS_DIR / "model_comparison.csv"
    predictions_path = OUTPUTS_DIR / "test_predictions.csv"
    report_path = OUTPUTS_DIR / "classification_report.txt"

    if not comparison_path.exists():
        raise FileNotFoundError(
            "model_comparison.csv not found. Run python src/train_models.py first."
        )

    comparison = pd.read_csv(comparison_path)
    print("Model comparison results:\n")
    print(comparison.to_string(index=False))

    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        print("\nSample test predictions:\n")
        print(predictions.head(10).to_string(index=False))

    if report_path.exists():
        print("\nClassification report:\n")
        print(report_path.read_text(encoding="utf-8"))

    print(
        "\nNote: the current project focuses on supervised churn prediction. "
        "Unsupervised comparison outputs are not generated in the present pipeline."
    )


if __name__ == "__main__":
    main()

