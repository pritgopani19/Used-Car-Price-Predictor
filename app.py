import streamlit as st
import pandas as pd
import joblib

# Load the dataset to extract unique dropdown values
df = pd.read_csv('Cleaned_Car.csv')

# Load trained model (pipeline)
pipe = joblib.load('LinearRegressionModel.pkl')  # make sure this is the trained pipeline

# Extract unique values from the dataset for dropdowns
car_names = sorted(df['name'].dropna().unique())
companies = sorted(df['company'].dropna().unique())
fuel_types = sorted(df['fuel_type'].dropna().unique())


# Streamlit UI
st.title("🚗 Used Car Price Predictor")
st.write("Enter the details of the car to estimate its selling price.")

# Input fields using real dataset values
name = st.selectbox('Car Model (name)', car_names)
company = st.selectbox('Brand (company)', companies)
year = st.number_input('Year of Manufacture', min_value=1990, max_value=2025, value=2020)
kms_driven = st.number_input('Kilometers Driven', min_value=0, step=500)
fuel_type = st.selectbox('Fuel Type', fuel_types)

# Predict button
if st.button('Predict Selling Price'):
    # Prepare input in correct format
    input_df = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    
    # Make prediction
    prediction = pipe.predict(input_df)
    st.success(f"Estimated Selling Price: ₹ {prediction[0]:,.2f}")
