from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests  # For fetching food details online
from pydantic import BaseModel

app = FastAPI()

# CORS setup (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = tf.keras.models.load_model("food101_model.h5")

# Food-101 class names (replace with your 101 labels)
class_names = ["apple_pie", "burger", "pizza", ...]  # Full list here: https://www.kaggle.com/datasets/kmader/food41

# Edamam API (for nutrition/recipes)
EDAMAM_API_KEY = "your_edamam_api_key"
EDAMAM_APP_ID = "your_edamam_app_id"

class PredictionResponse(BaseModel):
    food_name: str
    confidence: float
    calories: float
    recipe_url: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_from_camera(file: UploadFile = File(...)):
    try:
        # Step 1: Process the uploaded image
        img = Image.open(io.BytesIO(await file.read()))
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0

        # Step 2: Predict food class
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        food_name = class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        # Step 3: Fetch nutrition/recipe data from Edamam API
        nutrition_data = get_nutrition_data(food_name)
        recipe_data = get_recipe_data(food_name)

        return {
            "food_name": food_name,
            "confidence": confidence,
            "calories": nutrition_data.get("calories", 0),
            "recipe_url": recipe_data.get("url", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_nutrition_data(food_name: str):
    url = f"https://api.edamam.com/api/nutrition-data?app_id={EDAMAM_APP_ID}&app_key={EDAMAM_API_KEY}&ingr={food_name}"
    response = requests.get(url)
    return response.json().get("calories", 0)

def get_recipe_data(food_name: str):
    url = f"https://api.edamam.com/search?q={food_name}&app_id={EDAMAM_APP_ID}&app_key={EDAMAM_API_KEY}"
    response = requests.get(url)
    hits = response.json().get("hits", [])
    return hits[0]["recipe"] if hits else {"url": "#"}