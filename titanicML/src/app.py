import joblib
import numpy as np
import pandas as pd
import streamlit as st

from data_processing import basic_cleaning, load_data
from feature_engineering import add_engineered_features
from model_training import MODELS_DIR, prepare_data, train_and_select_model


@st.cache_resource
def get_or_train_model():
    model_path = MODELS_DIR / "best_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    model, _ = train_and_select_model()
    return model


def load_dataset():
    train_df, test_df = load_data()
    return basic_cleaning(add_engineered_features(train_df)), basic_cleaning(
        add_engineered_features(test_df)
    )


def main():
    st.set_page_config(page_title="Titanic ML Explorer", layout="wide")
    st.title("🚢 Titanic — Machine Learning from Disaster")
    st.write(
        "Explore the dataset, inspect model performance, and make survival predictions for custom passengers."
    )

    model = get_or_train_model()
    train_df, test_df = load_dataset()

    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Model Metrics", "🧮 Predict"])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(train_df.head())

        st.subheader("Survival by Sex")
        st.bar_chart(
            train_df.groupby("Sex")["Survived"].mean().rename("Survival Rate")
        )

        st.subheader("Survival by Passenger Class")
        st.bar_chart(
            train_df.groupby("Pclass")["Survived"].mean().rename("Survival Rate")
        )

    with tab2:
        st.subheader("Model Information")
        st.write(type(model).__name__)
        st.write("This model was trained using grid search over several algorithms.")
        st.info(
            "For detailed plots (confusion matrix, ROC), run the evaluation module directly in Python."
        )

    with tab3:
        st.subheader("Custom Passenger Prediction")

        col1, col2, col3 = st.columns(3)
        with col1:
            pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.slider("Age", 0, 80, 29)
        with col2:
            sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
            parch = st.number_input("Parents/Children Aboard (Parch)", 0, 6, 0)
            fare = st.number_input("Fare", 0.0, 600.0, 32.2)
        with col3:
            embarked = st.selectbox("Port of Embarkation (Embarked)", ["S", "C", "Q"])
            title = st.selectbox(
                "Title", ["Mr", "Mrs", "Miss", "Master", "Dr", "Rev", "Col", "Other"]
            )

        if st.button("Predict Survival"):
            input_df = pd.DataFrame(
                [
                    {
                        "Pclass": pclass,
                        "Sex": sex,
                        "Age": age,
                        "SibSp": sibsp,
                        "Parch": parch,
                        "Fare": fare,
                        "Embarked": embarked,
                        "Title": title,
                        "FamilySize": sibsp + parch + 1,
                    }
                ]
            )

            proba = model.predict_proba(input_df)[:, 1][0]
            pred = int(proba >= 0.5)

            st.metric("Predicted Survival Probability", f"{proba:.2%}")
            st.write("**Prediction:** Survived" if pred == 1 else "**Prediction:** Did not survive")


if __name__ == "__main__":
    main()


