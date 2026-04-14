import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide", page_icon="🫀")

# Load models and data
@st.cache_resource
def load_models():
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv("clean_heart_disease.csv")

try:
    model, scaler = load_models()
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    st.error("Model or data files not found. Please run `python data.py` and `python model.py` first.")
    data_loaded = False

st.title("🫀 Heart Disease Risk Predictor")
st.markdown("""
Welcome to the AI-powered Heart Disease Risk Predictor. 
Enter patient details in the sidebar to assess the risk of heart disease and view clinical recommendations.
""")

if data_loaded:
    tab1, tab2 = st.tabs(["🔮 Risk Prediction", "📊 Dashboard Insights"])
    
    with tab1:
        st.header("Patient Data Input")
        
        with st.sidebar:
            st.header("Medical Indicators")
            age = st.slider("Age", 20, 100, 50)
            sex_option = st.selectbox("Sex", ["Male", "Female"])
            sex = 1 if sex_option == "Male" else 0
            
            cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.slider("Serum Cholestoral (mg/dl)", 100, 400, 200)
            fbs_option = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            fbs = 1 if fbs_option == "Yes" else 0
            
            restecg = st.selectbox("Resting ECG", [0, 1, 2], help="0: Normal, 1: ST-T wave abnormality, 2: Probable/definite left ventricular hypertrophy")
            thalach = st.slider("Max Heart Rate Achieved", 70, 220, 150)
            exang_option = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            exang = 1 if exang_option == "Yes" else 0
            
            oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
            ca = st.slider("Number of Major Vessels (ca)", 0, 3, 0)
            thal = st.selectbox("Thalassemia (thal)", [1, 2, 3], help="1=Normal/Fixed, 3=Reversable Defect")
            
            st.markdown("---")
            predict_btn = st.button("Predict Risk", type="primary", use_container_width=True)
        
        if predict_btn:
            # Construct age_group matching data.py logic
            if age < 45:
                age_group = 0
            elif age < 60:
                age_group = 1
            else:
                age_group = 2
                
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, age_group]])
            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'age_group']
            
            # Predict
            scaled_features = scaler.transform(features)
            prob = model.predict_proba(scaled_features)[0][1]
            risk_percent = prob * 100
            
            st.subheader("Prediction Results")
            
            # Show Risk Output
            if prob < 0.3:
                risk_level = "Low Risk"
                color = "green"
            elif prob < 0.7:
                risk_level = "Medium Risk"
                color = "orange"
            else:
                risk_level = "High Risk"
                color = "red"
                st.toast("🚨 High Risk Alert Triggered!", icon="⚠️")
                
            st.markdown(f"### <span style='color:{color}'>{risk_level} ({risk_percent:.2f}%)</span>", unsafe_allow_html=True)
            
            # Health Recommendations
            st.subheader("Health Recommendations")
            if color == "green":
                st.success("Great! Keep maintaining a balanced diet, regular exercise, and routine check-ups.")
            elif color == "orange":
                st.warning("Moderate risk detected. Consider consulting a doctor to discuss lifestyle changes, specifically regarding cholesterol and blood pressure management.")
            else:
                st.error("URGENT: High risk detected! Please consult a cardiologist immediately for a comprehensive check-up.")
            
            st.markdown("---")
            
            # Explainable AI - SHAP
            st.subheader("Explainable AI (Feature Impact)")
            st.write("Understand which features contributed most to this prediction based on the trained algorithm.")
            
            try:
                # Provide a generic bar chart of internal tree feature importances
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    idx = np.argsort(importances)
                    fig, ax = plt.subplots(figsize=(8,5))
                    ax.barh(np.array(feature_names)[idx], importances[idx], color='teal')
                    ax.set_title("Overall Model Feature Importances")
                    st.pyplot(fig)
                else:
                    st.info("Visual explanation currently limited to tree-based models.")
            except Exception as e:
                st.write("Could not generate feature importance plot.")
                
    with tab2:
        st.header("Dataset Dashboard")
        st.write("Explore trends and correlations from historical patient data.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Age vs. Disease Presence")
            fig1 = plt.figure(figsize=(6,4))
            sns.histplot(data=df, x='age', hue='target', multiple="stack", palette="Set2")
            plt.title("Distribution of Age by Heart Disease")
            st.pyplot(fig1)
            
        with c2:
            st.subheader("Cholesterol vs. Blood Pressure")
            fig2 = plt.figure(figsize=(6,4))
            sns.scatterplot(data=df, x='chol', y='trestbps', hue='target', palette="coolwarm", alpha=0.7)
            plt.title("Cholesterol vs Resting BP colored by Disease")
            st.pyplot(fig2)
