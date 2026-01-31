import os
import io
import requests
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import time

# --- CONFIGURATION ---
MODEL_PATH = "results/checkpoints/best_model.keras"
IMG_SIZE = (224, 224)
CLASSES = ["Fresh", "Rotten"]
model = None

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model on startup and 'warm it up' so the first user request is fast.
    """
    global model
    print("⏳ Loading Model into Memory...")
    
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"❌ Model not found at {MODEL_PATH}")
        
    # 1. Load Model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded.")

    # 2. Warmup
    print("🔥 Warming up GPU kernels...")
    dummy_input = tf.zeros((1, 224, 224, 3))
    model.predict(dummy_input, verbose=0)
    print("🚀 API is Ready and Hot!")
    
    yield
    print("🛑 Shutting down...")

# --- APP DEFINITION ---
app = FastAPI(
    title="Fruit Quality SOTA API",
    description="A High-Performance Inference API for detecting Rotten vs. Fresh fruit using ConvNeXt Large.",
    version="1.0.0",
    lifespan=lifespan
)

# --- INPUT SCHEMA ---
class ImageUrlRequest(BaseModel):
    url: str = Field(
        ..., 
        description="Public URL of the fruit image", 
        example="https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Red_Apple.jpg/1200px-Red_Apple.jpg"
    )

@app.get("/", tags=["Health"])
def home():
    return {"status": "online", "docs_url": "/docs"}

@app.post("/predict", tags=["Inference"])
async def predict_fruit(request: ImageUrlRequest):
    """
    **Classify a Fruit Image**
    
    - Downloads image from URL
    - Resizes to 224x224
    - Runs inference on A100 GPU
    - Returns 'Fresh' or 'Rotten' with confidence score
    """
    global model
    
    start_time = time.time()
    
    # 1. Download
    try:
        response = requests.get(request.url, timeout=5)
        response.raise_for_status()
        image_bytes = response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")

    # 2. Preprocess
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_tensor = tf.expand_dims(img_array, 0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    # 3. Inference
    pred_probs = model.predict(img_tensor, verbose=0)
    
    # 4. Result
    score = pred_probs[0]
    result_class = CLASSES[np.argmax(score)]
    confidence = float(np.max(score))

    return {
        "prediction": result_class,
        "confidence": f"{confidence:.2%}",
        "raw_confidence": confidence,
        "inference_time": f"{time.time() - start_time:.3f}s"
    }