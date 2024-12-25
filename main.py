from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
# Cargar variables de entorno desde .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY no encontrada en el archivo .env")

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Middleware para validar API_KEY
@app.middleware("http")
async def api_key_validator(request: Request, call_next):
    api_key = request.headers.get("Authorization")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API_KEY no válida o no proporcionada")
    response = await call_next(request)
    return response

model = tf.keras.models.load_model('./modelo_entrenado')

@app.get("/")
async def home():
    return {"message": "Hello, World!"}

# End point que recibe una imagen como parámetro, hace un predict con el modelo convolucional de keras guardado con savemodel y retorna la clase predicha
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Leer la imagen directamente desde la memoria
    image_bytes = await file.read()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))  # Redimensionar a la entrada esperada por el modelo
    # Convertir la imagen en un array
    img_array = np.array(img, dtype=np.float32) / 255.0

    img_array = tf.expand_dims(img_array, 0)  # Agregar la dimensión del lote

    # Hacer la predicción
    predictions = model.predict(img_array)

    # Obtener la clase predicha
    predicted_class = np.argmax(predictions[0], axis=-1)
    
    # Diccionario de clases
    class_dict = {
        0: "rain",
        1: "snow",
        2: "sandstorm",
        3: "rime",
        4: "dew",
        5: "fogsmog",
        6: "frost",
        7: "hail",
        8: "glaze",
        9: "rainbow",
        10: "lightning"
    }
    
    # Crear el diccionario de salida con las probabilidades
    probabilities_dict = {class_dict[i]: float(predictions[0][i]) for i in range(len(predictions[0]))}

    clave_max = max(probabilities_dict, key=probabilities_dict.get)

    # Obtener el valor máximo
    valor_max = probabilities_dict[clave_max]

    return {
        "probability": valor_max,
        "predicted_class_name": clave_max,
        "all_probabilities": probabilities_dict
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)