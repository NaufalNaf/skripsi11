import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('klasifikasi_obesitas_svm.pkl')

# Load the scaler (assuming X_train_smote is defined elsewhere)
scaler = StandardScaler()

# Function to preprocess and predict
def preprocess_and_predict(gender, height, weight):
    # Convert gender to numeric (0 for Female, 1 for Male)
    gender_num = 0 if gender == 'Female' else 1

    # Preprocess the input data
    data = [[gender_num, height, weight]]
    
    # Fit scaler to training data (assuming X_train_smote is defined elsewhere)
    X_train_smote = ...  # Define your X_train_smote here
    scaler.fit(X_train_smote)

    # Transform the input data
    data_scaled = scaler.transform(data)

    # Predict with the model
    prediction = model.predict(data_scaled)[0]

    # Map prediction index to label
    index_labels = {
        0: 'Extremely Weak',
        1: 'Weak',
        2: 'Normal',
        3: 'Overweight',
        4: 'Obese',
        5: 'Extremely Obese'
    }

    # Return the prediction label
    return index_labels[prediction]

# Streamlit application title
st.title('Obesity Classification')

# Input fields for user
gender = st.selectbox('Gender', ['Female', 'Male'])
height = st.number_input('Height (cm)')
weight = st.number_input('Weight (kg)')

# Prediction button
if st.button('Classify'):
    # Get prediction
    prediction = preprocess_and_predict(gender, height, weight)
    st.write('Classification Result:', prediction)
