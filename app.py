import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image  # Añadir importación para Image

# Cargar el modelo
modelo_ruta = 'modelo_deteccion_edad.h5'
model = load_model(modelo_ruta)

st.title("Age Detection")

uploaded_file = st.file_uploader("Cargar una imagen", type=["jpg", "jpeg", "png"])

def extract_feature(img):
    features = []
    img = img.resize((128, 128), Image.ANTIALIAS)
    img = img_to_array(img)  
    img = np.expand_dims(img, axis=0) 
    return img

def age_range(edad_predicha):
    rango_inferior = edad_predicha - 2
    rango_superior = edad_predicha + 2
    print(f"Edad entre {rango_inferior} y {rango_superior} años.")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    sujeto = extract_feature(img)

    pred = model.predict(sujeto)
    pred_age = round(pred[0][0])
    print("Edad predicha:", pred_age)
    age_range(pred_age)

    st.image(img, caption="Imagen cargada", use_column_width=True)
