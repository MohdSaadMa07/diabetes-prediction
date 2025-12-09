import streamlit as st
import pandas as pd
import joblib
import numpy as np
import streamlit as st
st.title("Diabetes Prediction with Decision Tree")

@st.cache_resource
def load_model():
    return joblib.load('diabetes_model.pkl')

model = load_model()

# Input fields matching your dataset
pregnancies = st.number_input("Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 130, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 850, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 30.0)
pedigree = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5)
age = st.number_input("Age", 20, 90, 30)

inputs = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])

if st.button("Predict"):
    prediction = model.predict(inputs)[0]
    st.write("**Result:**" if prediction == 1 else "**Non-Diabetic**")
    prob = model.predict_proba(inputs)[0][1]
    st.write(f"Diabetes Probability: {prob:.2%}")
