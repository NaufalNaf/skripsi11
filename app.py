import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = joblib.load('klasifikasi_obesitas_svm.pkl')
scaler = StandardScaler()

st.title('Klasifikasi Obesitas')

gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
height = st.number_input('Tinggi Badan (cm)')
weight = st.number_input('Berat Badan (kg)')

if st.button('Klasifikasi'):
  data = [[gender, height, weight]]
  data_scaled = scaler.transform(data)
  prediction = model.predict(data_scaled)[0]
  index_labels = {
    0: 'Extremely Weak',
    1: 'Weak',
    2: 'Normal',
    3: 'Overweight',
    4: 'Obese',
    5: 'Extremely Obese'
  }
  st.write('Hasil Klasifikasi:', index_labels[prediction])
