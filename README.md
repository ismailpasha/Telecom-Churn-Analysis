# Telecom Churn Prediction

This project predicts whether a telecom customer is likely to churn using three machine learning models:

- Logistic Regression
- Random Forest
- XGBoost

The project is organized so you can work in a notebook for analysis and also run Python scripts for training and evaluation.

## Project Objective

The goal is to identify customers who are likely to leave a telecom service so the business can act early with retention strategies.

## Folder Structure

```text
Telecom_Churn_Prediction/
├── data/
│   ├── raw/
│   │   └── telecom_churn.csv
│   └── processed/
├── notebooks/
│   └── churn_prediction.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_models.py
│   └── evaluate_models.py
├── outputs/
│   ├── figures/
│   └── models/
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

Place your dataset here:

`data/raw/telecom_churn.csv`

Expected columns are typical churn fields such as:

- `customerID`
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `InternetService`
- `Contract`
- `MonthlyCharges`
- `TotalCharges`
- `Churn`

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How To Run

Train models:

```powershell
python src/train_models.py
```

Evaluate saved results:

```powershell
python src/evaluate_models.py
```

Start the notebook:

```powershell
jupyter notebook
```

Then open:

`notebooks/churn_prediction.ipynb`

## Expected Outputs

When training runs successfully, the project will create:

- `outputs/model_comparison.csv`
- `outputs/test_predictions.csv`
- `outputs/figures/confusion_matrix_best_model.png`
- `outputs/figures/feature_importance_best_model.png`

## Next Steps

- Add hyperparameter tuning
- Add SHAP interpretation for XGBoost
- Add a stronger business summary after final results

