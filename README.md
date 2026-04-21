# Telecom Churn Prediction

A Python machine learning project for predicting customer churn in telecom data and turning the predictions into business actions.

## What The Project Does

- Predicts whether a customer will churn.
- Compares three supervised models: Logistic Regression, Decision Tree, and Random Forest.
- Saves the best trained pipeline for later prediction.
- Generates feature importance, customer segmentation, and revenue-at-risk reports.
- Produces executive-level business insights for retention planning.

## Main Inputs

- Raw dataset: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Optional processed dataset: `data/processed/telco_churn_processed.csv`

## Main Outputs

- `outputs/model_comparison.csv`
- `outputs/test_predictions.csv`
- `outputs/classification_report.txt`
- `outputs/reports/model_comparison.csv`
- `outputs/reports/classification_report.txt`
- `outputs/reports/feature_importance.csv`
- `outputs/figures/feature_importance.png`
- `outputs/reports/customer_segments.csv`
- `outputs/reports/revenue_at_risk.csv`

## How To Run The Full Project

1. Install dependencies.
2. Run the training pipeline:
	- `python src/train_models.py`
3. View the saved outputs:
	- `python src/evaluate_models.py`
4. Run the top-level analysis flow:
	- `python main.py`

## Jupyter Notebook Run Order

Use the notebook in `notebooks/churn_prediction.ipynb` in this order:

1. Import libraries and set the project path.
2. Load the churn dataset.
3. Clean the data and engineer features.
4. Split data into train and test sets.
5. Build the preprocessing pipeline.
6. Train and compare models.
7. Save the best model and review metrics.
8. Generate feature importance.
9. Generate business insights.
10. Review the saved figures and CSV reports.

## Project Structure

- `src/config.py` centralizes paths and settings.
- `src/logger.py` configures console and file logging.
- `src/data_preprocessing.py` loads, cleans, and engineers features.
- `src/feature_engineering.py` prepares train/test data.
- `src/model_comparison.py` trains and evaluates models.
- `src/train_models.py` runs the full training pipeline.
- `src/predict.py` loads the saved model and generates predictions.
- `src/feature_importance.py` computes feature importance.
- `src/customer_segmentation.py` creates customer risk groups.
- `src/revenue_at_risk.py` estimates monthly revenue at risk.
- `src/business_insights.py` generates executive summaries.
- `src/evaluate_models.py` prints saved results.

## Features Included

- Data cleaning and missing value handling.
- Target encoding for churn.
- Engineered features such as `is_new_customer`, `avg_charge_per_month`, and `num_services`.
- Preprocessing for numeric and categorical columns.
- Model comparison using accuracy, precision, recall, F1, and ROC AUC.
- Best model persistence with `joblib`.
- Prediction probabilities and risk levels.
- Feature importance reporting.
- Customer segmentation and retention recommendations.
- Revenue-at-risk estimation.
- Business-level summary reporting.

## What Is Missing Or Weak

- The current project does not implement the unsupervised comparison outputs referenced in `src/evaluate_models.py`.
- The notebook is present, but it would still benefit from cleaner section titles and narrative text.
- There is only smoke-test coverage right now; more detailed unit tests would strengthen the project.
- The README can still be expanded with screenshots or sample result tables if you want a presentation-ready version.

## Requirements

The main packages are listed in `requirements.txt`.

## Quick Summary

This project takes telecom customer data, trains classification models to predict churn, saves the best model, and turns the prediction results into business actions such as customer segmentation, revenue-at-risk estimation, and retention recommendations.
