import streamlit as st
import pandas as pd
import joblib

# Title of the App
st.title("Titanic Survival Prediction by Ahmad Said")

# Load the trained model
model = joblib.load("best_xgboost_model.pkl")

# Sidebar for user input
st.sidebar.header("Passenger Information")

# User inputs
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Gender", ["Male", "Female"])
embarked = st.sidebar.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])
title = st.sidebar.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"])
familysize = st.sidebar.slider("Family Size", 1, 11, 1)
agegroup = st.sidebar.selectbox("Age Group", ["Child", "Teen", "Adult", "Senior"])
fareband = st.sidebar.selectbox("Fare Band", ["Low", "Mid", "High", "Very High"])

# Encoding user input
sex = 0 if sex == "Male" else 1
embarked = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]
title = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}[title]
agegroup = {"Child": 0, "Teen": 1, "Adult": 2, "Senior": 3}[agegroup]
fareband = {"Low": 0, "Mid": 1, "High": 2, "Very High": 3}[fareband]

# Create DataFrame for prediction
input_data = pd.DataFrame([[pclass, sex, embarked, title, familysize, agegroup, fareband]], 
                          columns=["Pclass", "Sex", "Embarked", "Title", "FamilySize", "AgeGroup", "FareBand"])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"
    st.success(f"Prediction: {result}")
