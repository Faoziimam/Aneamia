import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="anemia.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Prediksi Anemia Berdasarkan Citra Mata dan Hb")
st.write("Masukkan data warna mata dan kadar hemoglobin untuk memprediksi apakah seseorang mengalami anemia.")

# Input dari pengguna
gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
red_pixel = st.number_input("% Red Pixel", min_value=0.0, max_value=100.0, value=44.0)
green_pixel = st.number_input("% Green Pixel", min_value=0.0, max_value=100.0, value=29.0)
blue_pixel = st.number_input("% Blue Pixel", min_value=0.0, max_value=100.0, value=27.0)
hb = st.number_input("Kadar Hemoglobin (Hb)", min_value=0.0, max_value=20.0, value=12.0)

if st.button("Prediksi Anemia"):
    # Ubah gender menjadi numerik
    sex = 1 if gender == "Laki-laki" else 0

    # Siapkan input
    input_data = np.array([[sex, red_pixel, green_pixel, blue_pixel, hb]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    # Prediksi
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_label = np.argmax(prediction)
    result = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Hasil prediksi: **{result.upper()}**")
