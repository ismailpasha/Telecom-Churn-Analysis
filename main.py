from src.feature_engineering import FeatureEngineer
from src.model_comparison import ModelComparison
from src.feature_importance import FeatureImportance
from src.business_insights import BusinessInsights


def main() -> None:
    engineer = FeatureEngineer()

    df = engineer.load_dataset()

    X_train, X_test, y_train, y_test = engineer.prepare_dataset(df)

    comparison = ModelComparison()

    results = comparison.compare_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    comparison.print_results(results)
    comparison.save_results(results)
    comparison.save_best_model(results)

    importance = FeatureImportance()

    importance.generate(X_train.columns)

    insights = BusinessInsights()

    insights.generate_report(df)


if __name__ == "__main__":
    main()