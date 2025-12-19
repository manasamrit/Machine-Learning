import pandas as pd
from pathlib import Path
from typing import Tuple


DATA_DIR = Path(__file__).resolve().parents[1] / "raw"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data from the raw folder."""
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Handle simple missing values with reasonable defaults."""
    df = df.copy()

    # Fill Embarked with mode
    if "Embarked" in df.columns:
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Fill Age with median
    if "Age" in df.columns:
        df["Age"].fillna(df["Age"].median(), inplace=True)

    # Fill Fare with median
    if "Fare" in df.columns:
        df["Fare"].fillna(df["Fare"].median(), inplace=True)

    return df


__all__ = ["load_data", "basic_cleaning", "DATA_DIR"]


