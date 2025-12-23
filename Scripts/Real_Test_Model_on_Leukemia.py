import pickle
import pandas as pd

# loading model and scaler
with open("leukemia_final_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("leukemia_final_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# patient data frame (afzal taha)
# (must match your training columns)
new_patient = pd.DataFrame({
    'Age': [45],
    'Gender': [1], # Male
    'Country': [2],
    'WBC_Count': [12000],
    'RBC_Count': [4.2],
    'Platelet_Count': [150000],
    'Hemoglobin_Level': [10.5],
    'Bone_Marrow_Blasts': [65], # high blast count
    'Genetic_Mutation': [1],
    'Family_History': [0],
    'Smoking_Status': [0],
    'Alcohol_Consumption': [0],
    'Radiation_Exposure': [1],
    'Infection_History': [0],
    'BMI': [24.5],
    'Chronic_Illness': [0],
    'Immune_Disorders': [0],
    'Ethnicity': [1],
    'Socioeconomic_Status': [1],
    'Urban_Rural': [1]
})

# prediction
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)

# probability (how confident is the model?)
probability = model.predict_proba(new_patient)[0][1]

if prediction[0] == 1:
    print("Result: Leukemia Detected (Positive)")
    print(f"CONFIDENCE: {probability * 100:.2f}%")
else:
    print("Result: No Leukemia Detected (Negative)")
