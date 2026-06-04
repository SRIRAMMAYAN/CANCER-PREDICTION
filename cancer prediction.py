import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path(__file__).resolve().parent / "dataset.csv"


def load_data():
    try:
        data = pd.read_csv(DATA_PATH, engine="python")
        return data.fillna("Unknown")
    except FileNotFoundError:
        st.error(f"dataset.csv not found at {DATA_PATH}")
    except MemoryError:
        st.error("MemoryError: The file might be too large to fit into memory.")
    except pd.errors.ParserError as e:
        st.error(f"ParserError: {e}")
    except Exception as e:
        st.error(f"Error loading data: {e}")
    return None


def preprocess_data(data: pd.DataFrame):
    data = data.copy()
    data = data.fillna("Unknown")

    categorical_cols = [
        "Gender",
        "Symptoms",
        "Medical_History",
        "Ethnicity",
        "Lymph_Node_Involvement",
        "Generation_Report",
    ]
    label_encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str))
        label_encoders[col] = encoder

    target_encoder = LabelEncoder()
    data["Cancer_Type"] = target_encoder.fit_transform(data["Cancer_Type"].astype(str))
    y = np.asarray(data["Cancer_Type"].astype(np.int64))

    X = data.drop(["Patient_ID", "Cancer_Type"], axis=1, errors="ignore")
    scaler = StandardScaler().fit(X)
    return X, y, scaler, label_encoders, target_encoder


def train_cancer_type_model(data: pd.DataFrame):
    X, y, scaler, label_encoders, target_encoder = preprocess_data(data)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, scaler, label_encoders, target_encoder, X.columns


def determine_stage(size: float):
    if size < 2.0:
        return "Stage 1"
    if size < 3.0:
        return "Stage 2"
    if size < 4.0:
        return "Stage 3"
    return "Stage 4"


def recommend_medication(cancer_type: str, stage: str):
    meds = {
        "Melanoma": {
            "Stage 1": "Immunotherapy",
            "Stage 2": "Targeted Therapy",
            "Stage 3": "Chemotherapy",
            "Stage 4": "Palliative Care",
        },
        "Colorectal": {
            "Stage 1": "Surgery",
            "Stage 2": "Chemotherapy",
            "Stage 3": "Targeted Therapy",
            "Stage 4": "Palliative Care",
        },
    }
    return meds.get(cancer_type, {}).get(stage, "Standard care and monitoring")


def predict_life_expectancy(cancer_type: str, stage: str, with_med: bool = True):
    base_values = {
        "Melanoma": {"Stage 1": 10, "Stage 2": 8, "Stage 3": 5, "Stage 4": 2},
        "Colorectal": {"Stage 1": 12, "Stage 2": 10, "Stage 3": 7, "Stage 4": 3},
    }
    base = base_values.get(cancer_type, {}).get(stage)
    if base is None:
        return None
    return base if with_med else max(0, base - 2)


def plot_life_expectancy(cancer_type: str, stage: str):
    with_med = predict_life_expectancy(cancer_type, stage, with_med=True)
    without_med = predict_life_expectancy(cancer_type, stage, with_med=False)
    if with_med is None or without_med is None:
        st.warning("Life expectancy data not available for this cancer type and stage.")
        return

    df = pd.DataFrame(
        {"Years": [with_med, without_med]},
        index=["With Medication", "Without Medication"],
    )
    st.bar_chart(df)


def plot_cancer_type_increase():
    years = list(range(2015, 2021))
    cancer_types = ["Melanoma", "Colorectal"]
    increase_data = {cancer_type: np.random.randint(50, 150, len(years)) for cancer_type in cancer_types}
    st.line_chart(pd.DataFrame(increase_data, index=years))


def safe_transform(encoder: LabelEncoder, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        value = "Unknown"
    value = str(value)
    if value not in encoder.classes_:
        if "Unknown" in encoder.classes_:
            value = "Unknown"
        else:
            raise ValueError(f"Value '{value}' cannot be encoded with available labels")
    encoded = np.asarray(encoder.transform([value]), dtype=np.int64)
    return int(encoded[0])


def prediction_page(data: pd.DataFrame):
    st.header("Cancer Type Classification")
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    size = st.slider("Tumor Size (cm)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

    data = data.fillna("Unknown")
    gender_options = sorted(data["Gender"].astype(str).unique())
    symptoms_options = sorted(data["Symptoms"].astype(str).unique())
    history_options = sorted(data["Medical_History"].astype(str).unique())
    ethnicity_options = sorted(data["Ethnicity"].astype(str).unique())
    lymph_options = sorted(data["Lymph_Node_Involvement"].astype(str).unique())
    genetic_options = sorted(data["Generation_Report"].astype(str).unique())

    gender = st.selectbox("Gender", gender_options)
    symptoms = st.selectbox("Symptoms", symptoms_options)
    med_history = st.selectbox("Medical History", history_options)
    ethnicity = st.selectbox("Ethnicity", ethnicity_options)
    lymph_node = st.selectbox("Lymph Node Involvement", lymph_options)
    gen_report = st.selectbox("Genetic Report", genetic_options)

    if st.button("Predict"):
        try:
            model, scaler, encoders, target_encoder, features = train_cancer_type_model(data)
            input_data = pd.DataFrame(
                {
                    "Age": [age],
                    "Tumor_Size_cm": [size],
                    "Lymph_Node_Involvement": [safe_transform(encoders["Lymph_Node_Involvement"], lymph_node)],
                    "Symptoms": [safe_transform(encoders["Symptoms"], symptoms)],
                    "Medical_History": [safe_transform(encoders["Medical_History"], med_history)],
                    "Gender": [safe_transform(encoders["Gender"], gender)],
                    "Ethnicity": [safe_transform(encoders["Ethnicity"], ethnicity)],
                    "Generation_Report": [safe_transform(encoders["Generation_Report"], gen_report)],
                },
                columns=features,
            )

            prediction = model.predict(scaler.transform(input_data))
            prediction = np.asarray(prediction, dtype=np.int64).ravel()
            cancer_type = target_encoder.inverse_transform(prediction)[0]
            stage = determine_stage(size)

            st.success(f"Predicted Cancer Type: {cancer_type} - {stage}")
            st.info(f"Medication: {recommend_medication(cancer_type, stage)}")
            life_years = predict_life_expectancy(cancer_type, stage)
            if life_years is not None:
                st.info(f"Estimated Life Expectancy: {life_years} years")
            else:
                st.info("Life expectancy data not available.")

            plot_life_expectancy(cancer_type, stage)
            plot_cancer_type_increase()
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)


def main():
    st.title("Cancer Prediction System")
    data = load_data()
    if data is None:
        return
    prediction_page(data)


if __name__ == "__main__":
    main()
