import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Attempt to read the dataset with error handling
def load_data():
    try:
        # Attempt to read the CSV file
        data = pd.read_csv('dataset.csv', engine='python')
        return data
    except MemoryError:
        st.error("MemoryError: The file might be too large to fit into memory.")
        return None
    except pd.errors.ParserError as e:
        st.error(f"ParserError: {str(e)}")
        return None

data = load_data()

# Check if data is loaded successfully
if data is not None:
    # Preprocessing functions
    def preprocess_data():
        data.fillna(method='ffill', inplace=True)

        # Encode categorical variables
        label_encoders = {}
        for column in ['Gender', 'Symptoms', 'Medical_History', 'Ethnicity', 'Lymph_Node_Involvement', 'Cancer_Type', 'Generation_Report']:
            if column in data.columns:
                label_encoder = LabelEncoder()
                data[column] = label_encoder.fit_transform(data[column].astype(str))
                label_encoders[column] = label_encoder

        # Feature scaling
        scaler = StandardScaler()
        X = data.drop(['Patient_ID', 'Cancer_Type'], axis=1, errors='ignore')  # 'errors' param added to ignore missing columns
        X_scaled = scaler.fit_transform(X)
        return X, X_scaled, label_encoders, scaler

    # Model training for cancer type classification
    def train_cancer_type_model():
        X, X_scaled, label_encoders, scaler = preprocess_data()
        y = data['Cancer_Type']
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_scaled, y)
        feature_names = X.columns
        return rf_model, label_encoders, scaler, feature_names

    # Function to determine the cancer stage based on tumor size
    def determine_stage(tumor_size_cm):
        if 1 <= tumor_size_cm < 2:
            return "Stage 1"
        elif 2 <= tumor_size_cm < 3:
            return "Stage 2"
        elif 3 <= tumor_size_cm < 4:
            return "Stage 3"
        elif 4 <= tumor_size_cm <= 5:
            return "Stage 4"
        else:
            return "Stage Unknown"

    # Function to recommend medication based on cancer type and stage
    def recommend_medication(cancer_type, stage):
        medications = {
            "Melanoma": {"Stage 1": "Immunotherapy", "Stage 2": "Targeted Therapy", "Stage 3": "Chemotherapy", "Stage 4": "Palliative Care"},
            "Colorectal": {"Stage 1": "Surgery", "Stage 2": "Chemotherapy", "Stage 3": "Targeted Therapy", "Stage 4": "Palliative Care"},
            "Prostate": {"Stage 1": "Surgery", "Stage 2": "Radiation Therapy", "Stage 3": "Hormone Therapy", "Stage 4": "Chemotherapy"},
            "Breast": {"Stage 1": "Surgery", "Stage 2": "Radiation Therapy", "Stage 3": "Chemotherapy", "Stage 4": "Targeted Therapy"},
            "Ovarian": {"Stage 1": "Surgery", "Stage 2": "Chemotherapy", "Stage 3": "Targeted Therapy", "Stage 4": "Palliative Care"},
            "Lymphoma": {"Stage 1": "Chemotherapy", "Stage 2": "Radiation Therapy", "Stage 3": "Immunotherapy", "Stage 4": "Palliative Care"},
            "Pancreatic": {"Stage 1": "Surgery", "Stage 2": "Chemotherapy", "Stage 3": "Targeted Therapy", "Stage 4": "Palliative Care"},
            "Lung": {"Stage 1": "Surgery", "Stage 2": "Radiation Therapy", "Stage 3": "Chemotherapy", "Stage 4": "Targeted Therapy"},
        }
        return medications.get(cancer_type, {}).get(stage, "No specific medication found")

    # Function to predict life expectancy based on cancer type and stage
    def predict_life_expectancy(cancer_type, stage, with_medication=True):
        life_expectancy_with_medication = {
            "Melanoma": {"Stage 1": 10, "Stage 2": 8, "Stage 3": 5, "Stage 4": 2},
            "Colorectal": {"Stage 1": 12, "Stage 2": 10, "Stage 3": 7, "Stage 4": 3},
            "Prostate": {"Stage 1": 14, "Stage 2": 11, "Stage 3": 6, "Stage 4": 2},
            "Breast": {"Stage 1": 15, "Stage 2": 12, "Stage 3": 8, "Stage 4": 4},
            "Ovarian": {"Stage 1": 13, "Stage 2": 10, "Stage 3": 6, "Stage 4": 3},
            "Lymphoma": {"Stage 1": 16, "Stage 2": 13, "Stage 3": 8, "Stage 4": 3},
            "Pancreatic": {"Stage 1": 11, "Stage 2": 8, "Stage 3": 5, "Stage 4": 2},
            "Lung": {"Stage 1": 9, "Stage 2": 7, "Stage 3": 4, "Stage 4": 1},
        }

        life_expectancy_without_medication = {
            "Melanoma": {"Stage 1": 7, "Stage 2": 5, "Stage 3": 3, "Stage 4": 1},
            "Colorectal": {"Stage 1": 9, "Stage 2": 7, "Stage 3": 4, "Stage 4": 1},
            "Prostate": {"Stage 1": 10, "Stage 2": 8, "Stage 3": 4, "Stage 4": 1},
            "Breast": {"Stage 1": 11, "Stage 2": 9, "Stage 3": 5, "Stage 4": 2},
            "Ovarian": {"Stage 1": 9, "Stage 2": 7, "Stage 3": 4, "Stage 4": 2},
            "Lymphoma": {"Stage 1": 13, "Stage 2": 10, "Stage 3": 5, "Stage 4": 2},
            "Pancreatic": {"Stage 1": 8, "Stage 2": 6, "Stage 3": 3, "Stage 4": 1},
            "Lung": {"Stage 1": 6, "Stage 2": 4, "Stage 3": 2, "Stage 4": 0.5},
        }

        if with_medication:
            return life_expectancy_with_medication.get(cancer_type, {}).get(stage, "Data not available")
        else:
            return life_expectancy_without_medication.get(cancer_type, {}).get(stage, "Data not available")

    # Function to plot life expectancy with and without medication
    def plot_life_expectancy(cancer_type, stage):
        with_medication = predict_life_expectancy(cancer_type, stage, with_medication=True)
        without_medication = predict_life_expectancy(cancer_type, stage, with_medication=False)
        
        labels = ['With Medication', 'Without Medication']
        values = [with_medication, without_medication]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(labels, values, color=['green', 'red'])
        ax.set_ylabel('Life Expectancy (Years)')
        ax.set_title(f'Life Expectancy for {cancer_type} - {stage}')
        st.pyplot(fig)

    # Function to plot average survival years by cancer type
    def plot_survival_by_cancer_type():
        cancer_types = ["Melanoma", "Colorectal", "Prostate", "Breast", "Ovarian", "Lymphoma", "Pancreatic", "Lung"]
        stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
        survival_years = {cancer: [predict_life_expectancy(cancer, stage) for stage in stages] for cancer in cancer_types}

        fig, ax = plt.subplots(figsize=(10, 6))
        for cancer_type in cancer_types:
            ax.plot(stages, survival_years[cancer_type], marker='o', label=cancer_type)

        ax.set_xlabel('Stage')
        ax.set_ylabel('Average Survival Years')
        ax.set_title('Average Survival Years by Cancer Type and Stage')
        ax.legend()
        st.pyplot(fig)


# Function to plot cancer type increase over the years
    def plot_cancer_type_increase():
        years = list(range(2015, 2021))
        cancer_types = ["Melanoma", "Colorectal", "Prostate", "Breast", "Ovarian", "Lymphoma", "Pancreatic", "Lung"]
        
        # Simulate data for demonstration
        increase_data = {cancer_type: np.random.randint(50, 150, len(years)) for cancer_type in cancer_types}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for cancer_type in cancer_types:
            ax.plot(years, increase_data[cancer_type], marker='o', label=cancer_type)

        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Cases')
        ax.set_title('Increase in Cancer Types Over the Years')
        ax.legend()
        st.pyplot(fig)

    # Prediction Page
    def prediction_page():
        st.markdown(
            """
            <style>
            .stApp {
                background-image: url("https://img.freepik.com/premium-photo/pink-ribbon-symbol-breast-cancer-black-background_136401-375.jpg");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }
            .css-1v3fvcr {
                color: black;
                text-align: left;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.header("Cancer Type Classification")

        # Input fields
        age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)
        tumor_size = st.number_input("Tumor Size (cm)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        lymph_node = st.selectbox("Lymph Node Involvement", options=['Yes', 'No'])
        symptoms = st.multiselect("Symptoms", options=['Fatigue', 'Weight Loss', 'Pain', 'Skin Changes', 'Persistent Cough', 'Difficulty Swallowing', 'Change in Bowel Habits', 'Unusual Bleeding', 'Sores that do not Heal', 'Lump or Thickening'])
        medical_history = st.text_input("Medical History")
        gender = st.selectbox("Gender", options=['Male', 'Female'])
        ethnicity = st.text_input("Ethnicity")
        gen_Report = st.selectbox("Genetic Report", options=['Yes', 'No'])

        if st.button("Predict Cancer Type"):
            try:
                model, label_encoders, scaler, feature_names = train_cancer_type_model()

                # Encode selected symptoms as a combined string
                symptoms_combined = ' '.join(symptoms)
                
                # Prepare input data
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Tumor_Size_cm': [tumor_size],
                    'Lymph_Node_Involvement': [label_encoders['Lymph_Node_Involvement'].transform([lymph_node])[0]],
                    'Symptoms': [label_encoders['Symptoms'].transform([symptoms_combined])[0]] if symptoms_combined in label_encoders['Symptoms'].classes_ else [0],
                    'Medical_History': [label_encoders['Medical_History'].transform([medical_history])[0]] if medical_history in label_encoders['Medical_History'].classes_ else [0],
                    'Gender': [label_encoders['Gender'].transform([gender])[0]],
                    'Ethnicity': [label_encoders['Ethnicity'].transform([ethnicity])[0]] if ethnicity in label_encoders['Ethnicity'].classes_ else [0],
                    'Generation_Report': [label_encoders['Generation_Report'].transform([gen_Report])[0]],
                }, columns=feature_names)  # Ensure correct column names

                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)
                predicted_cancer_type = list(label_encoders['Cancer_Type'].inverse_transform([prediction[0]]))[0]
                stage = determine_stage(tumor_size)
                st.success(f"Predicted Cancer Type: {predicted_cancer_type}\tStage: {stage}")

                # Medication Recommendation
                medication = recommend_medication(predicted_cancer_type, stage)
                life_expectancy = predict_life_expectancy(predicted_cancer_type, stage)
                st.info(f"Recommended Medication: {medication}")
                st.info(f"Estimated Life Expectancy: {life_expectancy} years")

                # Plot Individual Report
                plot_life_expectancy(predicted_cancer_type, stage)
                plot_survival_by_cancer_type()
                plot_cancer_type_increase()
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

    if __name__ == "__main__":
        prediction_page()
