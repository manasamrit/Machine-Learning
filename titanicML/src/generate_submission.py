from pathlib import Path

import joblib
import pandas as pd

# Absolute imports so this script can be run from the project root via `python -m src.generate_submission`
from data_processing import basic_cleaning, load_data
from feature_engineering import add_engineered_features
from model_training import MODELS_DIR


def main():
    train_df, test_df = load_data()
    test_df_proc = basic_cleaning(add_engineered_features(test_df))

    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {model_path}. Train a model before generating submission."
        )

    model = joblib.load(model_path)

    # Align columns: drop Survived if accidentally present, ensure features subset
    feature_cols = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "FamilySize",
    ]
    for col in feature_cols:
        if col not in test_df_proc.columns:
            test_df_proc[col] = None

    X_test = test_df_proc[feature_cols]
    preds = model.predict(X_test)

    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": preds.astype(int)}
    )
    out_path = Path("submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved submission file to {out_path.resolve()}")


if __name__ == "__main__":
    main()


