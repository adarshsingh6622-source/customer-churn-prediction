import streamlit as st
import numpy as np
import pickle
import pandas as pd

features = pickle.load(open("features.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Customer Churn Prediction")

tenure = st.number_input("Tenure")
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

if st.button("Predict"):

    input_data = pd.DataFrame(np.zeros((1,len(features))),columns=features)
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly
    input_data["TotalCharges"] = total

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    if prediction[0]==1:
        st.error("Customer will Churn")
    else:
        st.success("Customer will Stay")