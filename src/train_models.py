from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint, uniform
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error, root_mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    r2_score,
    roc_curve,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data_preprocessing import build_preprocessor, prepare_dataframe, split_features_target


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"


def ensure_output_dirs() -> None:
    for path in [OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def build_models(preprocessor) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=8,
                        min_samples_split=5,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def get_search_spaces() -> dict[str, dict[str, object]]:
    return {
        "Logistic Regression": {
            "model__C": uniform(loc=0.05, scale=5.0),
            "model__solver": ["liblinear", "lbfgs"],
        },
        "Random Forest": {
            "model__n_estimators": randint(150, 501),
            "model__max_depth": randint(4, 15),
            "model__min_samples_split": randint(2, 16),
            "model__min_samples_leaf": randint(1, 7),
            "model__max_features": ["sqrt", "log2", None],
        },
        "XGBoost": {
            "model__n_estimators": randint(150, 501),
            "model__max_depth": randint(3, 10),
            "model__learning_rate": uniform(loc=0.02, scale=0.18),
            "model__subsample": uniform(loc=0.65, scale=0.35),
            "model__colsample_bytree": uniform(loc=0.65, scale=0.35),
            "model__reg_lambda": uniform(loc=0.5, scale=2.5),
        },
    }


def tune_model(
    model_name: str,
    pipeline: Pipeline,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
) -> tuple[Pipeline, dict[str, object], float]:
    search_spaces = get_search_spaces()
    param_space = search_spaces.get(model_name)

    if param_space is None:
        return pipeline, {}, float("nan")

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_space,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_fit, y_fit)

    return search.best_estimator_, search.best_params_, float(search.best_score_)


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_best_model.png", dpi=200)
    plt.close()


def plot_feature_importance(model_pipeline: Pipeline, model_name: str) -> None:
    fitted_preprocessor = model_pipeline.named_steps["preprocessor"]
    fitted_model = model_pipeline.named_steps["model"]

    if not hasattr(fitted_model, "feature_importances_"):
        return

    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = fitted_model.feature_importances_

    feature_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_df, x="importance", y="feature", color="teal")
    plt.title(f"Top Feature Importances - {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_best_model.png", dpi=200)
    plt.close()


def plot_supervised_comparison(results_df: pd.DataFrame) -> None:
    comparison_long = results_df.melt(
        id_vars="Model",
        value_vars=["CV_ROC_AUC_Mean", "Test_ROC_AUC", "Test_F1"],
        var_name="Metric",
        value_name="Score",
    )

    metric_labels = {
        "CV_ROC_AUC_Mean": "CV ROC-AUC",
        "Test_ROC_AUC": "Test ROC-AUC",
        "Test_F1": "Test F1",
    }
    comparison_long["Metric"] = comparison_long["Metric"].map(metric_labels)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=comparison_long, x="Model", y="Score", hue="Metric")
    plt.ylim(0, 1)
    plt.title("Supervised Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "supervised_model_comparison.png", dpi=200)
    plt.close()


def plot_roc_curves(y_true: pd.Series, model_probabilities: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(8, 6))
    for model_name, y_prob in model_probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={model_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.title("ROC Curves - Supervised Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves_all_models.png", dpi=200)
    plt.close()


def plot_precision_recall_curves(y_true: pd.Series, model_probabilities: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(8, 6))
    for model_name, y_prob in model_probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, linewidth=2, label=f"{model_name} (AUC={pr_auc:.3f})")

    plt.title("Precision-Recall Curves - Supervised Models")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curves_all_models.png", dpi=200)
    plt.close()


def tune_threshold_for_f1(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def plot_threshold_impact(results_df: pd.DataFrame) -> None:
    threshold_df = results_df[["Model", "Test_F1", "Test_F1_Tuned"]].melt(
        id_vars="Model",
        value_vars=["Test_F1", "Test_F1_Tuned"],
        var_name="F1_Type",
        value_name="Score",
    )
    threshold_df["F1_Type"] = threshold_df["F1_Type"].map(
        {"Test_F1": "F1 @ 0.50", "Test_F1_Tuned": "F1 @ Tuned Threshold"}
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=threshold_df, x="Model", y="Score", hue="F1_Type")
    plt.ylim(0, 1)
    plt.title("Threshold Tuning Impact on F1 Score")
    plt.xlabel("Model")
    plt.ylabel("F1 score")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_tuning_f1_comparison.png", dpi=200)
    plt.close()


def run_unsupervised_analysis(X: pd.DataFrame, y: pd.Series, best_supervised_auc: float) -> pd.DataFrame:
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    k_values = list(range(2, 9))
    cluster_metrics_rows = []
    best_k = None
    best_silhouette = -1.0

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X_processed)
        score = silhouette_score(X_processed, cluster_labels)
        inertia = kmeans.inertia_
        davies = davies_bouldin_score(X_processed, cluster_labels)
        calinski = calinski_harabasz_score(X_processed, cluster_labels)

        cluster_metrics_rows.append(
            {
                "k": k,
                "silhouette_score": score,
                "inertia": inertia,
                "davies_bouldin": davies,
                "calinski_harabasz": calinski,
            }
        )

        if score > best_silhouette:
            best_silhouette = score
            best_k = k

    cluster_metrics_df = pd.DataFrame(cluster_metrics_rows)
    cluster_metrics_df.to_csv(OUTPUTS_DIR / "unsupervised_clustering_metrics.csv", index=False)
    cluster_metrics_df[["k", "silhouette_score"]].to_csv(
        OUTPUTS_DIR / "unsupervised_silhouette_scores.csv", index=False
    )

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=cluster_metrics_df, x="k", y="silhouette_score", marker="o")
    plt.title("KMeans Silhouette Score by Number of Clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "unsupervised_silhouette_scores.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=cluster_metrics_df, x="k", y="inertia", marker="o")
    plt.title("KMeans Elbow Curve (Inertia)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "unsupervised_elbow_curve.png", dpi=200)
    plt.close()

    final_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    final_labels = final_kmeans.fit_predict(X_processed)

    if issparse(X_processed):
        reducer = TruncatedSVD(n_components=2, random_state=42)
        components = reducer.fit_transform(X_processed)
    else:
        reducer = PCA(n_components=2, random_state=42)
        components = reducer.fit_transform(X_processed)

    cluster_plot_df = pd.DataFrame(
        {
            "Component_1": components[:, 0],
            "Component_2": components[:, 1],
            "Cluster": final_labels,
            "Churn": y.values,
        }
    )

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=cluster_plot_df,
        x="Component_1",
        y="Component_2",
        hue="Cluster",
        palette="tab10",
        alpha=0.75,
        s=25,
        linewidth=0,
    )
    plt.title(f"Unsupervised Clusters Projection (k={best_k})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "unsupervised_clusters_projection.png", dpi=200)
    plt.close()

    cluster_churn = (
        pd.DataFrame({"Cluster": final_labels, "Churn": y.values})
        .groupby("Cluster", as_index=False)
        .agg(churn_rate=("Churn", "mean"), customers=("Churn", "size"))
        .sort_values("churn_rate", ascending=False)
    )
    cluster_churn.to_csv(OUTPUTS_DIR / "unsupervised_cluster_churn_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=cluster_churn, x="Cluster", y="churn_rate", color="coral")
    plt.ylim(0, 1)
    plt.title("Churn Rate by Unsupervised Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Churn rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "unsupervised_cluster_churn_rate.png", dpi=200)
    plt.close()

    comparison_df = pd.DataFrame(
        {
            "Technique": ["Best Supervised (ROC-AUC)", "Best Unsupervised (Silhouette)"],
            "Score": [best_supervised_auc, best_silhouette],
        }
    )
    comparison_df.to_csv(OUTPUTS_DIR / "supervised_vs_unsupervised_comparison.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison_df, x="Technique", y="Score", palette=["teal", "orange"])
    plt.ylim(0, 1)
    plt.title("Best Supervised vs Unsupervised Score")
    plt.xlabel("Approach")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "supervised_vs_unsupervised_comparison.png", dpi=200)
    plt.close()

    return cluster_churn


def main() -> None:
    sns.set_style("whitegrid")
    ensure_output_dirs()

    df = prepare_dataframe()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train,
    )

    preprocessor = build_preprocessor(X)
    models = build_models(preprocessor)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    best_model_name = None
    best_score = -1.0
    best_pipeline = None
    best_predictions = None
    model_probabilities = {}
    best_threshold = 0.5

    for name, pipeline in models.items():
        tuned_pipeline, best_params, tuning_cv_score = tune_model(name, pipeline, X_fit, y_fit)

        cv_scores = cross_val_score(
            tuned_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=None,
        )

        tuned_pipeline.fit(X_fit, y_fit)
        val_prob = tuned_pipeline.predict_proba(X_val)[:, 1]
        tuned_threshold, _ = tune_threshold_for_f1(y_val, val_prob)

        tuned_pipeline.fit(X_train, y_train)
        y_prob = tuned_pipeline.predict_proba(X_test)[:, 1]
        y_pred_default = (y_prob >= 0.5).astype(int)
        y_pred_tuned = (y_prob >= tuned_threshold).astype(int)
        model_probabilities[name] = y_prob

        row = {
            "Model": name,
            "Tuning_CV_ROC_AUC": round(tuning_cv_score, 4),
            "Best_Params": str(best_params),
            "CV_ROC_AUC_Mean": round(cv_scores.mean(), 4),
            "CV_ROC_AUC_Std": round(cv_scores.std(), 4),
            "Best_Threshold": round(tuned_threshold, 2),
            "Test_Accuracy": round(accuracy_score(y_test, y_pred_default), 4),
            "Test_Precision": round(precision_score(y_test, y_pred_default), 4),
            "Test_Recall": round(recall_score(y_test, y_pred_default), 4),
            "Test_F1": round(f1_score(y_test, y_pred_default), 4),
            "Test_Accuracy_Tuned": round(accuracy_score(y_test, y_pred_tuned), 4),
            "Test_Precision_Tuned": round(precision_score(y_test, y_pred_tuned), 4),
            "Test_Recall_Tuned": round(recall_score(y_test, y_pred_tuned), 4),
            "Test_F1_Tuned": round(f1_score(y_test, y_pred_tuned), 4),
            "Test_ROC_AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Test_Log_Loss": round(log_loss(y_test, y_prob), 4),
            "Test_Brier_Score": round(brier_score_loss(y_test, y_prob), 4),
            "Test_MSE_Prob": round(mean_squared_error(y_test, y_prob), 4),
            "Test_RMSE_Prob": round(root_mean_squared_error(y_test, y_prob), 4),
            "Test_MAE_Prob": round(mean_absolute_error(y_test, y_prob), 4),
            "Test_R2_Prob": round(r2_score(y_test, y_prob), 4),
        }
        results.append(row)

        if row["Test_ROC_AUC"] > best_score:
            best_score = row["Test_ROC_AUC"]
            best_model_name = name
            best_pipeline = tuned_pipeline
            best_threshold = tuned_threshold
            best_predictions = pd.DataFrame(
                {
                    "Actual": y_test.reset_index(drop=True),
                    "Predicted_Default_0_50": pd.Series(y_pred_default),
                    "Predicted_Tuned": pd.Series(y_pred_tuned),
                    "Predicted_Probability": pd.Series(y_prob),
                }
            )

    results_df = pd.DataFrame(results).sort_values("Test_ROC_AUC", ascending=False)
    results_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    plot_supervised_comparison(results_df)
    plot_roc_curves(y_test, model_probabilities)
    plot_precision_recall_curves(y_test, model_probabilities)
    plot_threshold_impact(results_df)

    if best_predictions is not None:
        best_predictions.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)

    if best_pipeline is not None and best_model_name is not None:
        with open(MODELS_DIR / "best_model.pkl", "wb") as model_file:
            pickle.dump(best_pipeline, model_file)

        y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        y_pred_tuned = (y_prob >= best_threshold).astype(int)
        plot_confusion_matrix(y_test, y_pred_tuned, best_model_name)
        plot_feature_importance(best_pipeline, best_model_name)

        report = classification_report(y_test, y_pred_tuned)
        (OUTPUTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    run_unsupervised_analysis(X, y, best_score)

    print("Training finished.")
    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved results to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()

