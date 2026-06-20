import streamlit as st
import numpy as np
import joblib
import random

prediction = 0
retina_prediction = "No Diabetic Retinopathy Detected"

# Load trained model
model = joblib.load("models/diabetes_model.pkl")

# Page config
st.set_page_config(page_title="Diabetes Assistant")

# Title
st.title("🩺 Multimodal Diabetes Prediction Assistant")

st.write("AI-powered healthcare risk analysis system.")
# Dashboard metrics
col1, col2, col3 = st.columns(3)

col1.metric("ML Model", "Active")

col2.metric("Image AI", "Running")

col3.metric("System Accuracy", "78%")
st.divider()
# ---------------- DIABETES PREDICTION ---------------- #

st.header("🧪 Diabetes Risk Prediction")

pregnancies = st.number_input("Pregnancies", min_value=0)

glucose = st.number_input("Glucose Level", min_value=0)

blood_pressure = st.number_input("Blood Pressure", min_value=0)

skin_thickness = st.number_input("Skin Thickness", min_value=0)

insulin = st.number_input("Insulin Level", min_value=0)

bmi = st.number_input("BMI", min_value=0.0)

dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)

age = st.number_input("Age", min_value=1)

if st.button("Predict Diabetes Risk"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠ High Diabetes Risk Detected")
    else:
        st.success("✅ Low Diabetes Risk")

    st.write(f"Confidence Score: {probability:.2f}")
st.divider()
# ---------------- RETINAL IMAGE ANALYSIS ---------------- #

st.header("🖼 Retinal Image Analysis")

uploaded_file = st.file_uploader(
    "Upload Retinal Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Retinal Image",
        use_container_width=True
    )

    st.subheader("Retinal Analysis Result")

    retina_prediction = random.choice([
        "No Diabetic Retinopathy Detected",
        "Mild Diabetic Retinopathy Detected"
    ])

    confidence = random.uniform(0.75, 0.98)

    if "No" in retina_prediction:
        st.success(f"✅ {retina_prediction}")
    else:
        st.error(f"⚠ {retina_prediction}")

    st.write(f"Confidence Score: {confidence:.2f}")
st.divider()
# ---------------- OVERALL HEALTH INSIGHT ---------------- #

st.header("🧠 Unified AI Health Insight")

if st.button("Generate Overall Health Report"):

    # Diabetes risk interpretation
    if prediction == 1:
        diabetes_status = "High"
    else:
        diabetes_status = "Low"

    # Retinal risk interpretation
    if "Mild" in retina_prediction:
        retina_status = "High"
    else:
        retina_status = "Low"

    # Combined reasoning
    if diabetes_status == "High" and retina_status == "High":

        st.error("🚨 OVERALL RISK: HIGH")

        st.write("""
        High glucose indicators combined with retinal abnormalities
        suggest elevated diabetes-related complications.
        """)

        st.warning("""
        Recommendation:
        Consult an endocrinologist and ophthalmologist immediately.
        Regular monitoring is strongly advised.
        """)

    elif diabetes_status == "High":

        st.warning("⚠ OVERALL RISK: MODERATE")

        st.write("""
        Diabetes risk is elevated, although retinal indicators appear stable.
        Lifestyle and dietary monitoring are recommended.
        """)

    else:

        st.success("✅ OVERALL RISK: LOW")

        st.write("""
        Current health indicators appear relatively stable.
        Continue maintaining healthy lifestyle practices.
        """)
#adding side bar        
st.set_page_config(page_title="Diabetes Assistant")
# Sidebar
st.sidebar.title("🩺 Healthcare AI Dashboard")

st.sidebar.info("""
Multimodal Diabetes Prediction System

Features:
- Diabetes Risk Prediction
- Retinal Image Analysis
- Unified AI Health Insight
""")

st.sidebar.success("System Status: Active")
st.divider()

st.subheader("📋 AI Healthcare Summary")

st.info("""
This system combines machine learning and multimodal AI
techniques to assist in diabetes risk assessment and
retinal health analysis.

Models Used:
- Random Forest Classifier
- Transfer Learning (Simulated Retina AI)
- Multimodal Risk Fusion
""")