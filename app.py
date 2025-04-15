# Import Libraries
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import streamlit as st

# Load Models
with open('transfomration_pipeline.pkl', 'rb') as file:
     transfomration_pipeline = pickle.load(file)

print(transfomration_pipeline)
# transfomration_pipeline = joblib.load("transfomration_pipeline.pkl")

model = load_model('model.h5')

# Create the app
st.title("Heart Attack Risk Prediction using Artificial Neural Network (ANN)")

# User input
age = st.slider('Age', 18, 92)
gender = st.selectbox('Sex', ['Male', 'Female'])
Cholesterol = st.number_input('Cholesterol', min_value=0, max_value=400, value=200)
Heart_Rate = st.number_input('Heart Rate', min_value=0, max_value=120, value=80)
Diabetes = st.selectbox('Diabetes', [0, 1])
Family_History = st.selectbox('Family History', [0, 1])
Smoking = st.selectbox('Smoking', [0, 1])
Obesity = st.selectbox('Obesity', [0, 1])
Alcohol_Consumption = st.selectbox('Alcohol Consumption', [0, 1])
Exercise_Hours_Per_Week = st.number_input('Exercise Hours Per Week', min_value=0, max_value=40, value=5)
Diet = st.selectbox('Diet', ['Average', 'Unhealthy', 'Healthy'])
Previous_Heart_Problems = st.selectbox('Previous Heart Problems', [0, 1])
Medication_Use = st.selectbox('Medication Use', [0, 1])
Stress_Level = st.slider('Stress Level', 0, 10)
Sedentary_Hours_Per_Day = st.number_input('Sedentary Hours Per Day', min_value=0, max_value=24, value=8)
Income = st.number_input('Income', min_value=0, max_value=1000000, value=134000)
BMI = st.number_input('BMI', min_value=0, max_value=50, value=20)
Triglycerides = st.number_input('Triglycerides', min_value=0, max_value=500, value=200)
Physical_Activity_Days_Per_Week = st.number_input('Physical Activity Days Per Week', min_value=0, max_value=7, value=1)
Sleep_Hours_Per_Day = st.number_input('Sleep Hours Per Day', min_value=0, max_value=24, value=4)
Country = st.selectbox('Country', ['Argentina', 'Canada', 'France', 'Thailand', 'Germany', 'Japan',
                                   'Brazil', 'South Africa', 'United States', 'Vietnam', 'China',
                                   'Italy', 'Spain', 'India', 'Nigeria', 'New Zealand', 'South Korea',
                                   'Australia', 'Colombia', 'United Kingdom'])
BP_High_Value = st.number_input('BP High Value', min_value=0, max_value=200, value=120)
BP_Low_Value = st.number_input('BP Low Value', min_value=0, max_value=120, value=80)
# Create input dictionary
input_data = {
    'Age': [age],
    'Sex': [gender],
    'Cholesterol': [Cholesterol],
    'Heart Rate': [Heart_Rate],
    'Diabetes': [Diabetes],
    'Family History': [Family_History],
    'Smoking': [Smoking],
    'Obesity': [Obesity],
    'Alcohol Consumption': [Alcohol_Consumption],
    'Exercise Hours Per Week': [Exercise_Hours_Per_Week],
    'Diet': [Diet],
    'Previous Heart Problems': [Previous_Heart_Problems],
    'Medication Use': [Medication_Use],
    'Stress Level': [Stress_Level],
    'Sedentary Hours Per Day': [Sedentary_Hours_Per_Day],
    'Income': [Income],
    'BMI': [BMI],
    'Triglycerides': [Triglycerides],
    'Physical Activity Days Per Week': [Physical_Activity_Days_Per_Week],
    'Sleep Hours Per Day': [Sleep_Hours_Per_Day],
    'Country': [Country],
    'BP High Value': [BP_High_Value],
    'BP Low Value': [BP_Low_Value]
}

# Transform the input data using the transformation pipeline
print(transfomration_pipeline)
input_df = pd.DataFrame(input_data)
transformed_input_df = transfomration_pipeline.transform(input_df)
print(transformed_input_df)

# # Predict the risk using the ANN model
# risk_prediction = model.predict(transformed_input_df)

# # Display the prediction result
# st.write(f"The predicted risk of heart attack is: {risk_prediction[0][0] * 100:.2f} %")
