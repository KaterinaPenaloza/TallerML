import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import cv2
from src.metricas import dice_coef, dice_loss
from matplotlib import pyplot as plt
def inferencia(imagen):

    # Ruta al archivo del modelo entrenado
    ruta_modelo = '/proyectos/taller_temporal/pesos/_epoca_01_val_loss_0.31_loss_0.30_val_dice_0.69_train_dice_0.70.h5'

    # Cargar el modelo
    modelo = load_model(ruta_modelo, custom_objects={'dice_coef': dice_coef, 'dice_loss': dice_loss})
    
    # Redimensionar la imagen a 224x224 y convertirla a escala de grises
    image_resized = cv2.resize(imagen, (224, 224))
    image_gray = tf.image.rgb_to_grayscale(image_resized)
    
    # Convertir a array numpy y normalizar
    image_gray = np.array(image_gray)
    image_gray = image_gray / 255.0
    
    image_gray = np.expand_dims(image_gray, axis=0)  # Agrega dimensión batch

    print("image_gray shape:", image_gray.shape) 

    # Realizar la predicción
    predicciones = modelo.predict(image_gray)
    predicciones = np.squeeze(predicciones)  
    #predicciones = (predicciones > 0).astype(np.uint8)  # Escalar a [0, 255] para mostrar la imagen

    print(np.unique(predicciones))
    return predicciones

def sepia(input_img):
    print("input_img shape:", input_img.shape)
    pred = inferencia(input_img)
    print("pred shape:", pred.shape)
    plt.imshow(pred, cmap="gray")
    plt.savefig("output.png")
    return pred

demo = gr.Interface(sepia, gr.Image(), "image")
demo.launch()
