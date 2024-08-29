import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load dataset
def load_data():
    try:
        return pd.read_csv('dataset.csv', engine='python')
    except MemoryError:
        st.error("MemoryError: The file might be too large to fit into memory.")
    except pd.errors.ParserError as e:
        st.error(f"ParserError: {str(e)}")
    return None

data = load_data()

if data is not None:
    def preprocess_data():
        data.fillna(method='ffill', inplace=True)
        label_encoders = {col: LabelEncoder().fit(data[col].astype(str)) for col in ['Gender', 'Symptoms', 'Medical_History', 'Ethnicity', 'Lymph_Node_Involvement', 'Cancer_Type', 'Generation_Report']}
        for col, le in label_encoders.items():
            data[col] = le.transform(data[col].astype(str))
        X = data.drop(['Patient_ID', 'Cancer_Type'], axis=1, errors='ignore')
        return X, StandardScaler().fit_transform(X), label_encoders

    def train_cancer_type_model():
        X, X_scaled, label_encoders = preprocess_data()
        rf_model = RandomForestClassifier(random_state=42).fit(X_scaled, data['Cancer_Type'])
        return rf_model, label_encoders, StandardScaler().fit(X), X.columns

    def determine_stage(size):
        return {1: "Stage 1", 2: "Stage 2", 3: "Stage 3", 4: "Stage 4"}.get(int(size), "Stage Unknown")

    def recommend_medication(cancer_type, stage):
        meds = {
            "Melanoma": {"Stage 1": "Immunotherapy", "Stage 2": "Targeted Therapy", "Stage 3": "Chemotherapy", "Stage 4": "Palliative Care"},
            "Colorectal": {"Stage 1": "Surgery", "Stage 2": "Chemotherapy", "Stage 3": "Targeted Therapy", "Stage 4": "Palliative Care"},
        }
        return meds.get(cancer_type, {}).get(stage, "No specific medication found")

    def predict_life_expectancy(cancer_type, stage, with_med=True):
        le_data = {
            "Melanoma": {"Stage 1": 10, "Stage 2": 8, "Stage 3": 5, "Stage 4": 2},
            "Colorectal": {"Stage 1": 12, "Stage 2": 10, "Stage 3": 7, "Stage 4": 3},
        }
        return le_data.get(cancer_type, {}).get(stage, "Data not available")

    def plot_life_expectancy(cancer_type, stage):
        with_med = predict_life_expectancy(cancer_type, stage, with_med=True)
        without_med = predict_life_expectancy(cancer_type, stage, with_med=False)
        st.bar_chart(pd.DataFrame({'Years': [with_med, without_med]}, index=['With Medication', 'Without Medication']))

    def plot_cancer_type_increase():
        years = list(range(2015, 2021))
        cancer_types = ["Melanoma", "Colorectal"]
        increase_data = {cancer_type: np.random.randint(50, 150, len(years)) for cancer_type in cancer_types}
        st.line_chart(pd.DataFrame(increase_data, index=years))

    def prediction_page():
        st.header("Cancer Type Classification")
        age, size = st.slider("Age", 0, 100, 30), st.slider("Tumor Size (cm)", 0.0, 5.0, 2.0)
        lymph_node, gender = st.selectbox("Lymph Node Involvement", ['Yes', 'No']), st.selectbox("Gender", ['Male', 'Female'])
        symptoms, med_history, ethnicity, gen_report = st.multiselect("Symptoms", ['Fatigue', 'Pain']), st.text_input("Medical History"), st.text_input("Ethnicity"), st.selectbox("Genetic Report", ['Yes', 'No'])

        if st.button("Predict"):
            try:
                model, encoders, scaler, features = train_cancer_type_model()
                input_data = pd.DataFrame({
                    'Age': [age], 'Tumor_Size_cm': [size],
                    'Lymph_Node_Involvement': [encoders['Lymph_Node_Involvement'].transform([lymph_node])[0]],
                    'Symptoms': [encoders['Symptoms'].transform([' '.join(symptoms)])[0]],
                    'Medical_History': [encoders['Medical_History'].transform([med_history])[0]],
                    'Gender': [encoders['Gender'].transform([gender])[0]],
                    'Ethnicity': [encoders['Ethnicity'].transform([ethnicity])[0]],
                    'Generation_Report': [encoders['Generation_Report'].transform([gen_report])[0]],
                }, columns=features)
                prediction = model.predict(scaler.transform(input_data))
                cancer_type = encoders['Cancer_Type'].inverse_transform(prediction)[0]
                stage = determine_stage(size)
                st.success(f"Predicted Cancer Type: {cancer_type} - {stage}")
                st.info(f"Medication: {recommend_medication(cancer_type, stage)}")
                st.info(f"Life Expectancy: {predict_life_expectancy(cancer_type, stage)} years")
                plot_life_expectancy(cancer_type, stage)
                plot_cancer_type_increase()
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if __name__ == "__main__":
        prediction_page()
