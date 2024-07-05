
import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('klasifikasi_obesitas_svm.pkl')

# Define the app
def main():
    st.title('Klasifikasi Obesitas')

    # Get input from the user
    gender = st.selectbox('Jenis Kelamin', ['Pria', 'Wanita'])
    height = st.number_input('Tinggi Badan (cm)', min_value=0.0, max_value=300.0)
    weight = st.number_input('Berat Badan (kg)', min_value=0.0, max_value=300.0)

    # Preprocess the input
    data = {'Gender': [gender], 'Height': [height], 'Weight': [weight]}
    df = pd.DataFrame(data)

    # Make a prediction
    if st.button('Klasifikasi'):
        prediction = model.predict(df)[0]
        st.write('Hasil Klasifikasi:', prediction)

# Run the app
if __name__ == '__main__':
    main()
