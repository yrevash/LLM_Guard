import requests
import tempfile
import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import cv2

app = FastAPI(
    title="Content Moderation API",
    description="API to check content safety using Qwen3-VL-2B model"
)

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# Load model and processor at startup
print(f"Loading model {MODEL_NAME} from HuggingFace...")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Model loaded successfully!")

SAFETY_PROMPT = """Analyze this content for safety and content moderation. 
Provide your response in the following exact format:

Rating: [safe/unsafe]
Category: [category code]

Category codes:
O1 - Hate, Humiliation, Harassment
O2 - Violence, Harm, or Cruelty
O3 - Sexual Content
O4 - Nudity Content
O5 - Criminal Planning
O6 - Weapons or Substance Abuse
O7 - Self-Harm
O8 - Animal Cruelty
O9 - Disasters or Emergencies
10 - None applying (Safe content)

Be concise and accurate in your assessment."""

TEXT_SAFETY_PROMPT = """Analyze this text for safety and content moderation:

TEXT: "{text}"

Provide your response in the following exact format:

Rating: [safe/unsafe]
Category: [category code]

Category codes:
O1 - Hate, Humiliation, Harassment
O2 - Violence, Harm, or Cruelty
O3 - Sexual Content
O4 - Nudity Content
O5 - Criminal Planning
O6 - Weapons or Substance Abuse
O7 - Self-Harm
O8 - Animal Cruelty
O9 - Disasters or Emergencies
10 - None applying (Safe content)

Be concise and accurate in your assessment."""


# ---------- Request/Response Models ----------
class ImageURLRequest(BaseModel):
    post_id: str
    image_url: str


class TextRequest(BaseModel):
    post_id: str
    text: str


class VideoURLRequest(BaseModel):
    post_id: str
    video_url: str


class SafetyData(BaseModel):
    post_id: str
    rating: str
    category: str


class SafetyResponse(BaseModel):
    success: bool
    data: SafetyData
    message: str


# ---------- Helper Functions ----------
def parse_model_response(response_text: str) -> dict:
    """Parse the model response to extract rating and category."""
    result = {
        "rating": "unknown",
        "category": "NA"
    }
    
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if line_lower.startswith('rating:'):
            rating_value = line.split(':', 1)[1].strip().lower()
            if 'unsafe' in rating_value:
                result["rating"] = "unsafe"
            elif 'safe' in rating_value:
                result["rating"] = "safe"
            else:
                result["rating"] = rating_value
                
        elif line_lower.startswith('category:'):
            category_value = line.split(':', 1)[1].strip().upper()
            # Extract category code (O1-O9 or NA)
            for cat in ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "10"]:
                if cat in category_value:
                    result["category"] = cat
                    break
    
    return result


def analyze_image(image: Image.Image) -> dict:
    """Analyze image using Qwen3-VL model."""
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": SAFETY_PROMPT}
                ]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"Model response: {response_text}")
        return parse_model_response(response_text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")


def analyze_text(text: str) -> dict:
    """Analyze text using Qwen3-VL model."""
    try:
        # Create a blank image for text-only analysis
        blank_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        
        prompt = TEXT_SAFETY_PROMPT.format(text=text)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": blank_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"Model response: {response_text}")
        return parse_model_response(response_text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")


def load_image_from_url(url: str) -> Image.Image:
    """Download and load image from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")


def extract_video_frames(video_path: str, num_frames: int = 5) -> list:
    """Extract frames from video for analysis."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError("Unable to read video frames")
    
    frame_indices = [int(total_frames * i / num_frames) for i in range(num_frames)]
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    cap.release()
    return frames


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"status": "active", "model": MODEL_NAME}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_NAME}


# Image Endpoints
@app.post("/classify/image", response_model=SafetyResponse)
async def classify_image(
    post_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Classify uploaded image for safety."""
    allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    
    result = analyze_image(image)
    return SafetyResponse(
        success=True,
        data=SafetyData(post_id=post_id, **result),
        message="success"
    )


@app.post("/classify/image/url", response_model=SafetyResponse)
async def classify_image_url(request: ImageURLRequest):
    """Classify image from URL for safety."""
    image = load_image_from_url(request.image_url)
    result = analyze_image(image)
    return SafetyResponse(
        success=True,
        data=SafetyData(post_id=request.post_id, **result),
        message="success"
    )


# Text Endpoint
@app.post("/classify/text", response_model=SafetyResponse)
async def classify_text_content(request: TextRequest):
    """Classify text content for safety."""
    result = analyze_text(request.text)
    return SafetyResponse(
        success=True,
        data=SafetyData(post_id=request.post_id, **result),
        message="success"
    )


# Video Endpoints
@app.post("/classify/video", response_model=SafetyResponse)
async def classify_video(
    post_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Classify uploaded video for safety (analyzes key frames)."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files are supported.")
    
    temp_video_path = None
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp.flush()
            temp_video_path = tmp.name
        
        frames = extract_video_frames(temp_video_path, num_frames=5)
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Analyze each frame, return unsafe if any frame is unsafe
        for frame in frames:
            result = analyze_image(frame)
            if result["rating"] == "unsafe":
                return SafetyResponse(
                    success=True,
                    data=SafetyData(post_id=post_id, **result),
                    message="success"
                )
        
        # All frames are safe
        return SafetyResponse(
            success=True,
            data=SafetyData(post_id=post_id, rating="safe", category="NA"),
            message="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video classification failed: {str(e)}")
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except:
                pass


@app.post("/classify/video/url", response_model=SafetyResponse)
async def classify_video_url(request: VideoURLRequest):
    """Classify video from URL for safety."""
    temp_video_path = None
    
    try:
        response = requests.get(request.video_url, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(response.content)
            temp_video_path = temp_file.name
        
        frames = extract_video_frames(temp_video_path, num_frames=5)
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        for frame in frames:
            result = analyze_image(frame)
            if result["rating"] == "unsafe":
                return SafetyResponse(
                    success=True,
                    data=SafetyData(post_id=request.post_id, **result),
                    message="success"
                )
        
        return SafetyResponse(
            success=True,
            data=SafetyData(post_id=request.post_id, rating="safe", category="NA"),
            message="success"
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video classification failed: {str(e)}")
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8102)
