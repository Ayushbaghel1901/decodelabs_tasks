import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('LR_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')


st.title("Heart Disease Prediction")    
st.markdown("Provide your details")

age = st.slider("Age", 10,100,40)
sex = st.selectbox("Sex", ['M','F'])
cp = st.selectbox("Chest pain type", ['ATA','NAP','TA','ASY'])
resting_bp = st.number_input("Resting Blood Pressure", 80,200,120)
cholestrol = st.number_input("Cholestrol", 100,600,200)
FastingBS = st.selectbox("Fasting Blood Sugar", ["Y","N"])
resting_ecg = st.selectbox("Resting ECG",["Normal","ST","LVH"])
MaxHR = st.slider("Max Heart Rate", 70,220,150)
exercise_angina = st.selectbox("Exercise Induced Angina",["Y","N"])
oldpeak = st.slider("Old Peak", 0,6,2)
ST_slope = st.selectbox("ST Slope", ["Up","Down","Flat"])



if st.button("Predict"):
    # 1. Map all inputs directly to the expected columns
    data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholestrol,
        "FastingBS": 1 if FastingBS == "Y" else 0,
        "MaxHR": MaxHR,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "M" else 0,
        "ChestPainType_ATA": 1 if cp == "ATA" else 0,
        "ChestPainType_NAP": 1 if cp == "NAP" else 0,
        "ChestPainType_TA": 1 if cp == "TA" else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0,
        "ST_Slope_Flat": 1 if ST_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if ST_slope == "Up" else 0,
    }
    input_df = pd.DataFrame([data])

    # 2. Scale only the 5 numeric columns
    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # 3. Ensure columns match the expected order exactly
    final_df = input_df[expected_columns]

    # 4. Predict
    prediction = model.predict(final_df)

    if prediction[0] == 1:
        st.write("Heart disease detected")
    else:
        st.write("No heart disease detected")
