from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_DATA_PATH = RAW_DATA_DIR / "telecom_churn.csv"


def resolve_data_path(path: Path | None = None) -> Path:
    if path is not None:
        return path

    if DEFAULT_DATA_PATH.exists():
        return DEFAULT_DATA_PATH

    telco_candidates = sorted(RAW_DATA_DIR.glob("*Telco*Churn*.csv"))
    if telco_candidates:
        return telco_candidates[0]

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    raise FileNotFoundError(
        f"Dataset not found in {RAW_DATA_DIR}. Place a churn CSV file in data/raw/."
    )


def load_data(path: Path | None = None) -> pd.DataFrame:
    data_path = resolve_data_path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Place a churn CSV file in data/raw/."
        )
    return pd.read_csv(data_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "Churn" in df.columns:
        if df["Churn"].dtype == "object":
            df["Churn"] = df["Churn"].str.strip().map({"Yes": 1, "No": 0})

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "tenure" in df.columns:
        df["is_new_customer"] = (df["tenure"] < 12).astype(int)
    else:
        df["is_new_customer"] = 0

    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["avg_charge_per_month"] = np.where(
            df["tenure"].fillna(0) > 0,
            df["TotalCharges"].fillna(0) / df["tenure"].replace(0, np.nan),
            0,
        )
        df["avg_charge_per_month"] = df["avg_charge_per_month"].replace([np.inf, -np.inf], 0).fillna(0)
    else:
        df["avg_charge_per_month"] = 0

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    active_cols = [col for col in service_cols if col in df.columns]

    def count_services(row: pd.Series) -> int:
        count = 0
        for col in active_cols:
            value = str(row[col]).strip().lower()
            if value not in {"no", "no internet service", "no phone service", "nan"}:
                count += 1
        return count

    df["num_services"] = df.apply(count_services, axis=1) if active_cols else 0
    return df


def prepare_dataframe(path: Path | None = None) -> pd.DataFrame:
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' was not found in the dataset.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

