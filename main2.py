import os
import io
import cv2
import torch
import tempfile
import requests
from PIL import Image
from io import BytesIO
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    pipeline,
    AutoImageProcessor,
    AutoModelForImageClassification
)
import uvicorn


app = FastAPI(
    title="Unified Safety Classification API",
    description="NSFW + Harmful Content Classification using Transformers & LLMs",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORIES = {
    "O1: Hate, Humiliation, Harassment": "Hate, harassment, or humiliation",
    "O2: Violence, Harm, or Cruelty": "Violence, harm, or cruelty",
    "O3: Sexual Content": "Sexual content",
    "O4: Nudity Content": "Nudity",
    "O5: Criminal Planning": "Criminal planning",
    "O6: Weapons or Substance Abuse": "Weapons or substance abuse",
    "O7: Self-Harm": "Self-harm",
    "O8: Animal Cruelty": "Animal cruelty",
    "O9: Disasters or Emergencies": "Disasters or emergencies",
    "NA: None applying": "Safe content"
}

generation_args = {
    "max_new_tokens": 10,
    "do_sample": False,
    "temperature": 0.1,
    "use_cache": True,
}

llava_model = None
llava_processor = None
classifier = None
nsfw_processor = None
nsfw_model = None


# Standard Response Model
class StandardResponse(BaseModel):
    success: bool
    data: Optional[Any]
    message: str


class URLRequest(BaseModel):
    post_id: str
    url: HttpUrl


class SafetyResponse(BaseModel):
    post_id: str
    rating: str
    category: str


class TextRequest(BaseModel):
    post_id: str
    text: str


def format_category(category: str) -> str:
    """Convert category from 'O1' format to '1' format, or keep 'NA' as is."""
    if category.startswith("O") and len(category) == 2 and category[1].isdigit():
        return category[1]
    return category


@app.on_event("startup")
def load_models():
    global llava_model, llava_processor, classifier, nsfw_model, nsfw_processor

    print("Loading LLavaGuard model...")
    llava_model_id = "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf"
    llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        llava_model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)

    llava_processor = AutoProcessor.from_pretrained(llava_model_id)

    print("Loading BART classifier...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if device.type == "cuda" else -1
    )
    
    print("Loading NSFW model...")
    nsfw_processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
    nsfw_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection").to(device)


def predict_nsfw(image: Image.Image):
    inputs = nsfw_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = nsfw_model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = nsfw_model.config.id2label[predicted_class_idx]
    confidence = logits.softmax(-1)[0][predicted_class_idx].item()
    return label, confidence


def get_llava_output(image, text=""):
    prompt = "Describe any potential harmful content in the image/text. Do not assume."
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt + "\n\n" + text}
        ]
    }]
    prompt_text = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = llava_processor(text=prompt_text, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = llava_model.generate(**inputs, **generation_args)
    return llava_processor.decode(output[0], skip_special_tokens=True).strip()


def classify_safety(explanation: str) -> Dict[str, Any]:
    result = classifier(explanation, list(CATEGORIES.values()), multi_label=True)
    best_score = 0
    best_category = "NA"
    
    for label, score in zip(result["labels"], result["scores"]):
        if label == "Safe content" and score > 0.6:
            return {"rating": "safe", "category": "NA"}
        elif score > best_score and score > 0.4:
            best_score = score
            for key, value in CATEGORIES.items():
                if value == label:
                    best_category = key.split(":")[0]
                    break
    
    formatted_category = format_category(best_category)
    
    return {
        "rating": "unsafe" if formatted_category != "NA" else "safe",
        "category": formatted_category
    }


def classify_image_internal(image: Image.Image):
    explanation = get_llava_output(image)
    safety = classify_safety(explanation)
    return safety


@app.get("/")
def root():
    return {
        "success": True,
        "data": {},
        "message": "Unified NSFW + Safety API is running"
    }


@app.get("/health")
def health_check():
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "models_loaded": all([llava_model, classifier, nsfw_model])
        },
        "message": "health_check_successfully"
    }


@app.get("/categories")
def get_categories():
    simplified_categories = {}
    for key, value in CATEGORIES.items():
        if key.startswith("O"):
            simplified_key = key[1]
            simplified_categories[simplified_key] = value
        else:
            simplified_categories["NA"] = value
    
    return {
        "success": True,
        "data": {
            "categories": simplified_categories
        },
        "message": "categories_reterived_successfully"
	    }


@app.post("/classify/image")
async def classify_image_url(req: URLRequest):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        response = requests.get(req.url, headers=headers)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        result = classify_image_internal(image)
        
        return {
            "success": True,
            "data": {
                "post_id": req.post_id,
                "rating": result["rating"],
                "category": result["category"]
            },
            "message": "image_classification_successfull"
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "message": "failed_to_load_images"
        }


@app.post("/classify/text")
async def classify_text(text_req: TextRequest):
    try:
        dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        explanation = get_llava_output(dummy_image, text_req.text)
        result = classify_safety(explanation)
        
        return {
            "success": True,
            "data": {
                "post_id": text_req.post_id,
                "rating": result["rating"],
                "category": result["category"]
            },
            "message": "text_classification_successfull"
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "message": "failed_to_classify_text"
        }


@app.post("/classify/video")
async def classify_video(req: URLRequest):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Referer": "https://qoneqt.com", 
            "Accept": "video/mp4,video/*;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        
        response = requests.get(req.url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp.flush()

            cap = cv2.VideoCapture(tmp.name)
            success, frame = cap.read()
            cap.release()
            os.unlink(tmp.name)

            if not success:
                return {
                    "success": False,
                    "data": None,
                    "message": "failed_to_read_video_frame"
                }

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = classify_image_internal(pil_image)
            
            return {
                "success": True,
                "data": {
                    "post_id": req.post_id,
                    "rating": result["rating"],
                    "category": result["category"]
                },
                "message": "video_classification_successfull"
            }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "message": "failed_to_load_video"
        }


if __name__ == "__main__":
    uvicorn.run("main2:app", host="0.0.0.0", port=8102, reload=False)
