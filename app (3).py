# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import numpy as np

with open("scaler (1).pkl", "rb") as f:
    scaler = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Flood Risk Classification", layout="centered")

st.title("Flood Risk Classification App")
st.write("Enter the input values below to predict the flood risk level.")

feature_names = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
       'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
       'Siltation', 'AgriculturalPractices', 'Encroachments',
       'IneffectiveDisasterPreparedness', 'DrainageSystems',
       'CoastalVulnerability', 'Landslides', 'Watersheds',
       'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
       'InadequatePlanning', 'PoliticalFactors']

user_inputs = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    user_inputs.append(value)

input_array = np.array(user_inputs).reshape(1, -1)

# Scale inputs
scaled_input = scaler.transform(input_array)

# Prediction
if st.button("Predict Flood Risk"):
    prediction = model.predict(scaled_input)[0]
    if prediction < 0.3:
      remark = "green zone"
    elif prediction < 0.6:
      remark = "yellow zone"
    else:
      remark = "red zone"
    st.write(f'flood chances prediction: {prediction}, we remark it as {remark}.')

