######## Script para generar csv de entrenamiento

import os
import pandas as pd

# Definir las carpetas de imágenes
carpetas = ['venenosa', 'no_venenosa']
etiquetas = [1, 0]

# Listas para almacenar las rutas y etiquetas
imagenes_paths = []
etiquetas_list = []

# Recorrer las carpetas y agregar las imágenes a las listas
for carpeta, etiqueta in zip(carpetas, etiquetas):
    for imagen in os.listdir(carpeta):
        if imagen.endswith(".png") or imagen.endswith(".jpg"):
            imagenes_paths.append(f'{carpeta}/{imagen}')
            etiquetas_list.append(etiqueta)

# Crear un DataFrame y guardarlo como CSV
df_train = pd.DataFrame({
    'ruta_imagen': imagenes_paths,
    'etiqueta': etiquetas_list
})

df_train.to_csv('patrones_entrenamiento.csv', index=False)
print("Archivo CSV de patrones de entrenamiento creado.")





######## Script para generar csv de prueba

import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf

# Definir las rutas de las imágenes de prueba
test_paths = ['test/serpiente_prueba.png']  # Ajustar según sea necesario
etiquetas_reales = [1]  # Ajustar según sea necesario
predicciones = []

# Definir el tamaño de las imágenes
img_size = 200

# Cargar el modelo entrenado (asegúrate de que el modelo esté guardado y cargado correctamente)
model = tf.keras.models.load_model('modelo_serpientes.h5')

# Preprocesar y predecir las imágenes de prueba
for test_path in test_paths:
    img = Image.open(test_path).resize((img_size, img_size)).convert('L')
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    
    prediccion = model.predict(img)
    predicciones.append(np.argmax(prediccion[0]))

# Crear un DataFrame y guardarlo como CSV
df_test = pd.DataFrame({
    'ruta_imagen': test_paths,
    'etiqueta_real': etiquetas_reales,
    'prediccion': predicciones
})

df_test.to_csv('patrones_prueba.csv', index=False)
print("Archivo CSV de patrones de prueba creado.")
