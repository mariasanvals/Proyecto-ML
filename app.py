import streamlit as st
import joblib
import numpy as np

# Cargar el modelo y el escalador
model = joblib.load('lgbm_model.pkl')
scaler = joblib.load('scaler_lgbm.pkl')

# Título de la aplicación
st.title('Predicción de Burnout')

# Ingresar el nombre de la persona.
nombre = st.text_input('Nombre y apellidos de la persona')

# Seleccionar el género de la persona
genero = st.selectbox('Género', ['Hombre', 'Mujer'])

# Seleccionar la empresa de la persona
empresa = st.selectbox('Empresa', ['Servicio', 'Producción'])

# Seleccionar la opción de teletrabajo
teletrabajo = st.selectbox('¿Tiene opción de teletrabajo?', ['Sí', 'No'])

# Slider para la posición de la persona en la empresa
posicion = st.slider('Posición en la empresa', min_value=0, max_value=5, step=1)

# Slider para el allocation de la persona
allocation = st.slider('Asignación de tareas de la persona', min_value=1, max_value=10, step=1)

# Slider para la fatiga mental de la persona
fatiga_mental = st.slider('Fatiga mental de la persona', min_value=0, max_value=10, step=1)

# Diccionario para mapear las entradas a valores numéricos
genero_map = {'Hombre': 0, 'Mujer': 1}
empresa_map = {'Servicio': 0, 'Producción': 1}
teletrabajo_map = {'Sí': 1, 'No': 0}

# Convertir entradas a valores numéricos
genero_val = genero_map[genero]
empresa_val = empresa_map[empresa]
teletrabajo_val = teletrabajo_map[teletrabajo]

# Botón para hacer la predicción
if st.button('Predecir Burnout'):
    # Crear array con los valores de las características
    features = np.array([[genero_val, empresa_val, teletrabajo_val, posicion, allocation, fatiga_mental]])
    
    # Escalar las características
    features_scaled = scaler.transform(features)
    
    # Hacer la predicción
    prediccion = model.predict(features_scaled)
    
    # Mostrar el resultado
    st.write(f'La predicción de burnout para {nombre} es: {prediccion[0]:.4f}')
