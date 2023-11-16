from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import streamlit as st

modelo_ruta = 'modelo_deteccion_edad.h5'
model = load_model(modelo_ruta)

gender_dict = {0: 'Male', 1: 'Female'}

st.title("Age Detection")

uploaded_file = st.file_uploader("Cargar una imagen", type=["jpg", "jpeg", "png"])

def extract_feature(image):
    img = image.resize((128, 128), Image.LANCZOS)
    img_array = img_to_array(img)
    img_array = img_array.mean(axis=-1, keepdims=True)  # Convertir a escala de grises
    img_array = img_array / 255.0  # Normalizar valores al rango [0, 1]
    return img_array

def age_range(edad_predicha):
    rango_inferior = max(0, edad_predicha - 2)
    rango_superior = edad_predicha + 2
    st.info(f"Edad entre {rango_inferior} y {rango_superior} años.")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    sujeto = extract_feature(img)
    sujeto = sujeto.reshape((1, 128, 128, 1))  # Agregar dimensión para el modelo
    pred = model.predict(sujeto)
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    st.write("Predicted Gender:", pred_gender)
    st.write("Predicted Age:", pred_age)
    age_range(pred_age)

    st.image(img, caption="Imagen cargada", use_column_width=True)
