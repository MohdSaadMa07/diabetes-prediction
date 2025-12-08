import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load('diabetes_model.pkl')

st.title("Diabetes Prediction App")
preg = st.number_input("Pregnancies")
gluc = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
ins = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):
    features = np.array([[preg, gluc, bp, skin, ins, bmi, dpf, age]])
    pred = model.predict(features)
    st.success(f"Prediction: {'Diabetic' if pred[0]==1 else 'Non-Diabetic'}")
