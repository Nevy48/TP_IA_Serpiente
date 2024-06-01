import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Crea las carpetas si no existen
os.makedirs('venenosa', exist_ok=True)
os.makedirs('no_venenosa', exist_ok=True)
os.makedirs('test', exist_ok=True)

# Mostrar cuántas imágenes tengo de cada categoría
print(f"Imágenes en 'venenosa': {len(os.listdir('venenosa'))}")
print(f"Imágenes en 'no_venenosa': {len(os.listdir('no_venenosa'))}")

# Define las categorías y etiquetas
categorias = ['venenosa', 'no_venenosa']
imagenes = []
labels = []

img_size = 200
label_map = {categorias[i]: i for i in range(len(categorias))}

# Carga y preprocesa las imágenes
for categoria in categorias:
    img_dir = './' + categoria
    for archivo in os.listdir(img_dir):
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            img_path = os.path.join(img_dir, archivo)
            img = Image.open(img_path).resize((img_size, img_size)).convert('L')
            img = np.asarray(img, dtype=np.float32)
            imagenes.append(img)
            labels.append(label_map[categoria])

imagenes = np.array(imagenes)
labels = np.array(labels)

# Mostrar una imagen
plt.figure()
plt.imshow(imagenes[0], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()

# Normaliza las imágenes
imagenes = imagenes / 255.0

# Crea el modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(img_size, img_size)), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compila el modelo
model.compile(optimizer='adam',  
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Mostrar la estructura del modelo
model.summary()

#Entrenar el modelo
model.fit(imagenes, labels, epochs=10)  

# Guardar el modelo entrenado
model.save('modelo_serpientes.h5')

#Cargar una imagen de prueba
img_path = './test/serpiente_prueba.png'
if os.path.exists(img_path):
    im = Image.open(img_path).resize((img_size, img_size)).convert('L')
    im = np.asarray(im) / 255.0
    im = np.array([im])
    
    # Realizar predicciones
    predicciones = model.predict(im)
    categoria_predicha = categorias[np.argmax(predicciones[0])]
    print(f"La serpiente es: {categoria_predicha}")
else:
    print("No se encontró la imagen de prueba.")
