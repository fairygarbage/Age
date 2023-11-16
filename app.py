from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import streamlit as st

modelo_ruta = 'modelo_deteccion_edad.h5'
model = load_model(modelo_ruta)

gender_dict = {0:'Male', 1:'Female'}

st.title("Age Detection")

uploaded_file = st.file_uploader("Cargar una imagen", type=["jpg", "jpeg", "png"])

def extract_feature(image):
    features = []
    img = load_img(image, grayscale=True)
    img = img.resize((128, 128), Image.LANCZOS)
    img = np.array(img)
    features.append(img)

    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return img

def age_range(edad_predicha):
    rango_inferior = edad_predicha - 2
    rango_superior = edad_predicha + 2
    print(f"Edad entre {rango_inferior} y {rango_superior} años.")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    sujeto = extract_feature(img)
    sujeto = sujeto/255.0

    pred = model.predict(sujeto.reshape(1, 128, 128, 1))
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    print("Predicted Gender:", pred_gender)
    age_range(pred_age)

    st.image(img, caption="Imagen cargada", use_column_width=True)
