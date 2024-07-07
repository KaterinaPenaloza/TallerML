import os
from src.dataloder_taller import DataLoader
import json
from model.modelo import unet
from tensorflow import keras
from keras import optimizers
from wandb.integration.keras import WandbCallback
import tensorflow as tf
from src.metricas import dice_coef, dice_loss
import numpy as np
import wandb
import matplotlib.pyplot as plt

def run(config, train_generator, val_generator):
    wandb.init(settings=wandb.Settings(start_method="thread", console='off'), config=config, mode=config["modo"], project=config["project"])

    modelo = unet(input_size=(224, 224, 1))
    optim = optimizers.Adam(config["lr"])

    # Compile el modelo con el optimizador Adam, la función de pérdida dice_loss y la métrica dice_coef
    modelo.compile(optimizer=optim, loss=dice_loss, metrics=[dice_coef])

    # Imprime el resumen del modelo
    modelo.summary()

    #Wandb Callback
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(config["directorio_almacenado"], "modelo_{epoch:02d}.h5"),
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
    )

    # Invoque el método fit del modelo con los generadores de entrenamiento y validación, y el número de épocas
    modelo.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config["epocas"],
        callbacks=[WandbCallback(), checkpoint_callback]
    )
    
    return modelo

def plot_predictions(model, data_loader):
    X, y = next(iter(data_loader))  # Obtener un lote de datos
    predictions = model.predict(X)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(X[0, :, :, 0], cmap="gray")
    axes[0].set_title("Imagen Original")
    
    axes[1].imshow(y[0, :, :, 0], cmap="gray")
    axes[1].set_title("Etiqueta Verdadera")
    
    axes[2].imshow(predictions[0, :, :, 0], cmap="gray")
    axes[2].set_title("Predicción del Modelo")
    
    overlay = X[0, :, :, 0] * 0.5 + predictions[0, :, :, 0] * 0.5
    axes[3].imshow(overlay, cmap="gray")
    axes[3].set_title("Superposición")
    
    for ax in axes:
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    config = {
        "epocas": 1,
        "batch_size": 1,
        "lr": 0.01,
        "modo": "online",
        "project": "taller_ml",
        "directorio_almacenado": "pesos"
    }
    
    if not os.path.exists(config["directorio_almacenado"]):
        os.mkdir(config["directorio_almacenado"])

    id_dataset = "C:\\Users\\K4tz3\\Desktop\\ML\\taller_remoto-main\\rutas_dataset\\id_dataset.json"
    rutas_dataset = "C:\\Users\\K4tz3\\Desktop\\ML\\taller_remoto-main\\rutas_dataset\\rutas_dataset.json"

    with open(rutas_dataset) as file:
        rutas_dataset = json.load(file)

    with open(id_dataset) as file:
        id_dataset = json.load(file)

    # Verificar y obtener los datos de entrenamiento y validación (test)
    train_ids = id_dataset.get("train")
    val_ids = id_dataset.get("test")

    # Instancia el DataLoader con el dataset de entrenamiento y validación para imágenes de 224x224 y de 1 canal
    train_loader = DataLoader(train_ids, rutas_dataset, image_size=(224, 224), batch_size=config["batch_size"], cantidad_canales=1, shuffle=True)
    val_loader = DataLoader(val_ids, rutas_dataset, image_size=(224, 224), batch_size=config["batch_size"], cantidad_canales=1, shuffle=False)

    modelo = run(config, train_loader, val_loader)

    # Graficar las curvas de aprendizaje
    # plot_learning_curves(history)  # Asegúrate de tener la función plot_learning_curves implementada

    # Graficar predicciones
    plot_predictions(modelo, val_loader)
