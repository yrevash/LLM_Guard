import os
import io
import cv2
import torch
import tempfile
import requests
from PIL import Image
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
import uvicorn

app = FastAPI(
    title="Unified Safety Classification API (Lightweight)",
    description="NSFW Content Classification using Falconsai ViT",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nsfw_processor = None
nsfw_model = None

class StandardResponse(BaseModel):
    success: bool
    data: Optional[Any]
    message: str

class URLRequest(BaseModel):
    post_id: str
    url: HttpUrl

class TextRequest(BaseModel):
    post_id: str
    text: str

@app.on_event("startup")
def load_models():
    global nsfw_model, nsfw_processor
    print(f"Loading Falconsai NSFW model on {device}...")
    
    model_id = "Falconsai/nsfw_image_detection"
    try:
        nsfw_processor = AutoImageProcessor.from_pretrained(model_id)
        nsfw_model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def classify_image_internal(image: Image.Image) -> Dict[str, Any]:
    """
    Classifies image using Falconsai/nsfw_image_detection.
    Returns standard 'rating' and 'category' format.
    """
    inputs = nsfw_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = nsfw_model(**inputs)
        
    # Apply Softmax to get probabilities
    probs = F.softmax(outputs.logits, dim=-1)
    
    # Get the predicted label and confidence
    predicted_class_idx = probs.argmax(-1).item()
    label = nsfw_model.config.id2label[predicted_class_idx]
    confidence = probs[0][predicted_class_idx].item()
    
    # Map Falconsai output (normal/nsfw) to your API standard
    # Falconsai: 'nsfw' usually means sexual content.
    if label == "nsfw" and confidence > 0.5: # Threshold check
        return {
            "rating": "unsafe",
            "category": "3",  # Mapped to O3 (Sexual Content)
            "confidence": confidence
        }
    else:
        return {
            "rating": "safe",
            "category": "NA",
            "confidence": confidence
        }

@app.get("/")
def root():
    return {
        "success": True,
        "message": "Falconsai NSFW Classification API is running"
    }

@app.get("/health")
def health_check():
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "model_loaded": nsfw_model is not None,
            "device": str(device)
        }
    }

@app.post("/classify/image")
async def classify_image_url(req: URLRequest):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        response = requests.get(req.url, headers=headers, timeout=10)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        result = classify_image_internal(image)
        
        return {
            "success": True,
            "data": {
                "post_id": req.post_id,
                "rating": result["rating"],
                "category": result["category"],
                "confidence": result["confidence"]
            },
            "message": "image_classification_successful"
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "success": False,
            "data": None,
            "message": "failed_to_process_image"
        }

@app.post("/classify/video")
async def classify_video(req: URLRequest):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://qoneqt.com", 
            "Accept": "*/*"
        }
        
        response = requests.get(req.url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp.flush()
            temp_filename = tmp.name

        cap = cv2.VideoCapture(temp_filename)
        success, frame = cap.read()
        cap.release()
        os.unlink(temp_filename)

        if not success:
            return {
                "success": False,
                "message": "failed_to_read_video_frame"
            }

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = classify_image_internal(pil_image)
        
        return {
            "success": True,
            "data": {
                "post_id": req.post_id,
                "rating": result["rating"],
                "category": result["category"],
                "confidence": result["confidence"]
            },
            "message": "video_classification_successful"
        }

    except Exception as e:
        return {
            "success": False,
            "message": "failed_to_load_video"
        }

@app.post("/classify/text")
async def classify_text(text_req: TextRequest):
    # Falconsai is Image-only. Returning error or stub.
    return {
        "success": False,
        "data": None,
        "message": "text_classification_not_supported_by_current_model"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8102, reload=False)