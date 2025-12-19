from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def extract_title(name: str) -> str:
    """Extract title from a passenger name."""
    if pd.isna(name):
        return "Unknown"
    parts = name.split(",")
    if len(parts) < 2:
        return "Unknown"
    remainder = parts[1]
    title_part = remainder.split(".")[0]
    return title_part.strip()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add title and family size features."""
    df = df.copy()
    df["Title"] = df["Name"].apply(extract_title)
    df["FamilySize"] = df.get("SibSp", 0) + df.get("Parch", 0) + 1
    return df


def build_preprocessing_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, list]:
    """Build a preprocessing pipeline for the given dataframe."""
    numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "Pclass"]
    categorical_features = ["Sex", "Embarked", "Title"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    feature_cols = numeric_features + categorical_features
    # Ensure all engineered columns exist (some may be missing in test/train at call time)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    return preprocessor, feature_cols


__all__ = ["extract_title", "add_engineered_features", "build_preprocessing_pipeline"]


