from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image

app = FastAPI()

model = tf.keras.models.load_model('./modelo_entrenado')

@app.get("/")
async def home():
    return {"message": "Hello, World!"}

#End point que recibe una imagen como parametro, hace un predict con el modelo convolucional de keras guardado con savemodel y retorna la clase predicha
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Leer la imagen directamente desde la memoria
    image_bytes = await file.read()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))  # Redimensionar a la entrada esperada por el modelo
    # Convertir la imagen en un array
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Agregar la dimensión del lote

    # Hacer la predicción
    predictions = model.predict(img_array)
    
    # Obtener la clase predicha
    predicted_class = np.argmax(predictions[0], axis=-1)
    return {"class": int(predicted_class)}




if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)