import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('klasifikasi_obesitas_svm.pkl')

# Define the application
def main():
    st.title('Klasifikasi Obesitas')

    # Get user input
    gender = st.selectbox('Jenis Kelamin', ['Pria', 'Wanita'])
    height = st.number_input('Tinggi Badan (cm)', min_value=100, max_value=250)
    weight = st.number_input('Berat Badan (kg)', min_value=20, max_value=200)

    # Encode the user input
    gender_encoded = 0 if gender == 'Pria' else 1

    # Create a DataFrame with the user input
    user_input = pd.DataFrame({
        'Gender': [gender_encoded],
        'Height': [height],
        'Weight': [weight]
    })

    # Scale the user input
    scaler = StandardScaler()
    scaler.fit(user_input)
    user_input_scaled = scaler.transform(user_input)

    # Predict the BMI category
    bmi_category = model.predict(user_input_scaled)[0]

    # Display the result
    st.write('Kategori BMI:', bmi_category)

if __name__ == '__main__':
    main()
