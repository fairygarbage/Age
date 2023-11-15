import streamlit as st
import pandas as pd

modelo_ruta = 'modelo_deteccion_edad.h5'
model = load_model(modelo_ruta)

st.title("Age Detection")

uploaded_file = st.file_uploader("Cargar una imagen", type=["jpg", "jpeg", "png"])

def extract_feature(image):
    features = []
    img = load_img(image, grayscale=True)
    img = img.resize((128, 128), Image.ANTIALIAS)
    img = np.array(img)
    features.append(img)

    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features
  
def age_range(edad_predicha):
    rango_inferior = edad_predicha - 2
    rango_superior = edad_predicha + 2

    print(f"Age between {rango_inferior} and {rango_superior} years:")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    sujeto = extract_feature(img)
    sujeto = sujeto/255.0

    pred = model.predict(sujeto.reshape(1, 128, 128, 1))
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    print("Predicted Gender:", pred_gender)
    age_range(pred_age)
    st.image(img)


