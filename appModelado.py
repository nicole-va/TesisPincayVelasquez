import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import io

# Cargar el modelo
model = CatBoostClassifier()
model.load_model('catboost_model.cbm')

# Crear una función para hacer predicciones y obtener probabilidades
def make_prediction(features):
    df = pd.DataFrame([features], columns=['NumeroOferentes', 'CategoriaPrincipal', 'Presupuesto', 'MontoOfertado', 'DuracionLicitacionDias', 'DuracionContratoDias'])
    prediction = model.predict(df)
    probabilidad = model.predict_proba(df)[0][1]  # Probabilidad de la clase 1
    return prediction[0], probabilidad, df

# Aplicación Streamlit
st.title('Recomendador de Participación en Licitaciones')
st.markdown('Esta aplicación predice la probabilidad de participación en un proceso licitatorio.')

# Crear entradas de usuario organizadas en columnas
col1, col2 = st.columns(2)

with col1:
    numero_oferentes = st.number_input('Número de Oferentes', min_value=0, help="Ingrese el número de oferentes en la licitación.")
    categoria_principal = st.selectbox('Categoría Principal', options=['goods', 'services', 'works'], help="Seleccione la categoría principal de la licitación.")
    presupuesto = st.number_input('Presupuesto', min_value=0.0, help="Ingrese el presupuesto total para la licitación.")

with col2:
    monto_ofertado = st.number_input('Monto Ofertado', min_value=0.0, help="Ingrese el monto ofertado.")
    duracion_licitacion_dias = st.number_input('Duración de la Licitación (Días)', min_value=0, help="Ingrese la duración de la licitación en días.")
    duracion_contrato_dias = st.number_input('Duración del Contrato (Días)', min_value=0, help="Ingrese la duración del contrato en días.")

# Convertir las entradas a valores numéricos
categoria_map = {'goods': 0, 'services': 1, 'works': 2}
categoria_num = categoria_map[categoria_principal]

if st.button('Hacer Predicción'):
    features = [numero_oferentes, categoria_num, presupuesto, monto_ofertado, duracion_licitacion_dias, duracion_contrato_dias]
    prediccion, probabilidad, df = make_prediction(features)
    
    st.subheader('Resultados')
    if probabilidad >= 0.75:
        st.success(f'**Recomendación:** Alta probabilidad de participación ({probabilidad:.2f}). Se recomienda participar en la licitación.')
    elif 0.5 <= probabilidad < 0.75:
        st.warning(f'**Recomendación:** Probabilidad moderada de participación ({probabilidad:.2f}). Considere participar si puede asumir ciertos riesgos.')
    else:
        st.error(f'**Recomendación:** Baja probabilidad de participación ({probabilidad:.2f}). Se recomienda no participar en la licitación.')
        
    # Visualización
    st.progress(int(probabilidad * 100))