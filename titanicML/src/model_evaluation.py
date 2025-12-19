from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Absolute imports so modules work when run from `src/` as a script
from data_processing import basic_cleaning, load_data
from feature_engineering import add_engineered_features, build_preprocessing_pipeline
from model_training import MODELS_DIR, build_models


FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def evaluate_models() -> Dict[str, float]:
    """Train models and save evaluation plots for the best model."""
    train_df, _ = load_data()
    train_df = basic_cleaning(add_engineered_features(train_df))

    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]

    preprocessor, _ = build_preprocessing_pipeline(X)
    models = build_models(preprocessor)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_model: Pipeline | None = None
    best_name = ""
    best_acc = -np.inf
    scores: Dict[str, float] = {}

    for name, (pipe, _) in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores[name] = acc
        if acc > best_acc:
            best_acc = acc
            best_model = pipe
            best_name = name

    if best_model is None:
        raise RuntimeError("No model could be trained for evaluation.")

    # Detailed metrics for best model
    y_proba = best_model.predict_proba(X_val)[:, 1]
    y_pred = best_model.predict(X_val)

    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {best_name}")
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y_val, y_proba)
    auc = roc_auc_score(y_val, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {best_name}")
    plt.legend(loc="lower right")
    roc_path = FIGURES_DIR / "roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    report = classification_report(y_val, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = FIGURES_DIR / "classification_report.csv"
    report_df.to_csv(report_path)

    # Save evaluated best model separately if desired
    eval_model_path = MODELS_DIR / "evaluated_best_model.joblib"
    joblib.dump(best_model, eval_model_path)

    scores["best_model_name"] = best_name  # type: ignore[assignment]
    scores["best_model_accuracy"] = best_acc
    scores["best_model_auc"] = float(auc)

    return scores


__all__ = ["evaluate_models", "FIGURES_DIR"]


