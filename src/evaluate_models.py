from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    comparison_path = OUTPUTS_DIR / "model_comparison.csv"
    predictions_path = OUTPUTS_DIR / "test_predictions.csv"
    report_path = OUTPUTS_DIR / "classification_report.txt"
    supervised_vs_unsupervised_path = OUTPUTS_DIR / "supervised_vs_unsupervised_comparison.csv"
    clustering_metrics_path = OUTPUTS_DIR / "unsupervised_clustering_metrics.csv"
    cluster_churn_path = OUTPUTS_DIR / "unsupervised_cluster_churn_summary.csv"

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

    if supervised_vs_unsupervised_path.exists():
        comparison = pd.read_csv(supervised_vs_unsupervised_path)
        print("\nBest supervised vs unsupervised comparison:\n")
        print(comparison.to_string(index=False))

    if clustering_metrics_path.exists():
        clustering_metrics = pd.read_csv(clustering_metrics_path)
        print("\nUnsupervised clustering metrics by k:\n")
        print(clustering_metrics.to_string(index=False))

    if cluster_churn_path.exists():
        cluster_churn = pd.read_csv(cluster_churn_path)
        print("\nCluster churn summary:\n")
        print(cluster_churn.to_string(index=False))


if __name__ == "__main__":
    main()

