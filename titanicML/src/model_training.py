from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Use absolute imports so running `streamlit run src/app.py` works without package context
from data_processing import basic_cleaning, load_data
from feature_engineering import add_engineered_features, build_preprocessing_pipeline


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def prepare_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_df, test_df = load_data()
    train_df = basic_cleaning(add_engineered_features(train_df))
    test_df = basic_cleaning(add_engineered_features(test_df))

    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]
    return X, y, test_df


def build_models(
    preprocessor: ColumnTransformer,
) -> Dict[str, Tuple[Pipeline, Dict]]:
    """Return models and their corresponding hyperparameter grids."""
    models: Dict[str, Tuple[Pipeline, Dict]] = {}

    log_reg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    log_reg_grid = {"clf__C": [0.1, 1.0, 10.0]}

    rf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", RandomForestClassifier(random_state=42)),
        ]
    )
    rf_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 5, 10],
    }

    gb = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", GradientBoostingClassifier(random_state=42)),
        ]
    )
    gb_grid = {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
    }

    svm = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", SVC(probability=True)),
        ]
    )
    svm_grid = {"clf__C": [0.5, 1.0, 2.0], "clf__kernel": ["rbf", "linear"]}

    models["log_reg"] = (log_reg, log_reg_grid)
    models["random_forest"] = (rf, rf_grid)
    models["gradient_boosting"] = (gb, gb_grid)
    models["svm"] = (svm, svm_grid)
    return models


def train_and_select_model() -> Tuple[Pipeline, Dict[str, float]]:
    X, y, _ = prepare_data()
    preprocessor, _ = build_preprocessing_pipeline(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models(preprocessor)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = "accuracy"

    best_model_name = None
    best_model = None
    best_val_score = -np.inf
    model_scores: Dict[str, float] = {}

    for name, (pipe, param_grid) in models.items():
        grid = GridSearchCV(
            pipe, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)

        y_pred_val = grid.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred_val)

        model_scores[name] = val_acc

        if val_acc > best_val_score:
            best_val_score = val_acc
            best_model = grid.best_estimator_
            best_model_name = name

    if best_model is None or best_model_name is None:
        raise RuntimeError("No best model selected.")

    # Save best model
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)

    model_scores["best_model"] = best_val_score
    model_scores["best_model_name"] = best_model_name  # type: ignore[assignment]

    return best_model, model_scores


__all__ = ["prepare_data", "build_models", "train_and_select_model", "MODELS_DIR"]


