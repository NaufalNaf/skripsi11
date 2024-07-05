import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

model = joblib.load('klasifikasi_obesitas_svm.pkl')
scaler = StandardScaler()

st.title('Klasifikasi Obesitas')

gender = st.selectbox('Jenis Kelamin', ['Pria', 'Wanita'])
height = st.number_input('Tinggi Badan (cm)', min_value=100, max_value=200)
weight = st.number_input('Berat Badan (kg)', min_value=20, max_value=200)

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
