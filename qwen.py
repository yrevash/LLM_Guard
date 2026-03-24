import asyncio
import gc
import json
import os
import re
import tempfile
from datetime import datetime
from io import BytesIO
from typing import Optional

import cv2
import requests
import torch
import pillow_avif
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, model_validator
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

# ─────────────────────────────────────────────
#  App & model setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Content Moderation API",
    description="API to check content safety using Qwen3-VL-2B model",
)

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

print(f"Loading model {MODEL_NAME} from HuggingFace...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Model loaded successfully!")

# Global async lock — prevents concurrent GPU access from corrupting state
model_lock = asyncio.Lock()

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────

SAFETY_PROMPT = """Analyze this content for safety and content moderation.
IMPORTANT: Read any text visible in the image carefully.

Provide your response in the following exact format (do NOT add any extra text):

Rating: [safe/unsafe]
Category: [category code]

Category codes:
1 - Hate, Humiliation, Harassment
2 - Violence, Harm, or Cruelty
3 - Sexual Content
4 - Nudity Content
5 - Criminal Planning
6 - Weapons or Substance Abuse
7 - Self-Harm
8 - Animal Cruelty
9 - Disasters or Emergencies
10 - None applying (Safe content)
11 - Spam, Scam, or Fraudulent Content

Be concise and accurate."""

TEXT_SAFETY_PROMPT = """Analyze this text for safety and content moderation:

TEXT: "{text}"

Provide your response in the following exact format (do NOT add any extra text):

Rating: [safe/unsafe]
Category: [category code]

Category codes:
1 - Hate, Humiliation, Harassment
2 - Violence, Harm, or Cruelty
3 - Sexual Content
4 - Nudity Content
5 - Criminal Planning
6 - Weapons or Substance Abuse
7 - Self-Harm
8 - Animal Cruelty
9 - Disasters or Emergencies
10 - None applying (Safe content)
11 - Spam, Scam, or Fraudulent Content

IMPORTANT rules for category 11 — mark as 11 ONLY if the text explicitly:
- Directly asks for money transfers ("send me ₹500", "pay me at", "donate to my account")
- Shows a completed payment transaction ("Received ₹300 from", "Payment successful", "Debited from account")
- Promotes referral/affiliate codes ("use my code ABC123", "sign up with my referral link")
- Is clearly a scam (fake lottery, fake giveaway, phishing, crypto pump-and-dump)

DO NOT mark as 11 if:
- Text is a normal social media post, caption, or opinion
- Text contains usernames/handles like @something
- Text mentions products, brands, food, drinks, or places casually
- Text is from a banner, poster, flyer, event notice, or announcement

When unsure, always choose category 10 (safe). False positives are worse than false negatives."""

# ─────────────────────────────────────────────
#  Spam keyword lists  (tightened to reduce false positives)
# ─────────────────────────────────────────────

# High-confidence transaction phrases — must include currency context
SPAM_TRANSACTION_PHRASES = [
    "received ₹", "paid ₹", "sent ₹",
    "₹ received", "₹ paid", "₹ sent",
    "received rs.", "paid rs.", "sent rs.",
    "received $", "paid $", "sent $",
    "payment successful", "payment received",
    "payment confirmed", "payment done",
    "money received", "money sent",
    "transaction successful", "transaction id",
    "transferred to bank", "deposited in your account",
    "credited to your account", "debited from your account",
    "balance withdrawal", "funds are expected",
    "upi transaction", "upi: ", "vpa: ",
]

# Currency regex — must be combined with a strong context keyword
SPAM_CURRENCY_RE = re.compile(
    r'[₹$]\s*\d+|\brs\.?\s*\d+|\d+\s*[₹$]|\d+\s*rs\b|\d+/-',
    re.IGNORECASE,
)

# Strong money-action verbs — only used alongside a currency hit
SPAM_STRONG_MONEY_VERBS = [
    "payment successful", "payment received", "money received",
    "transaction successful", "credited to your", "debited from your",
    "send to receive", "withdrawal successful",
    "obtained", "bonus", "receiving", "earned", "winning", "cashback", "payout",
]

# Payment platform names — only flagged when paired with a currency amount
SPAM_PLATFORM_NAMES = [
    "paytm", "phonepe", "google pay", "gpay", "bhim upi",
    "cashapp", "cash app", "venmo", "zelle",
]

# High-confidence solicitation / scam phrases (no ambiguity allowed)
SPAM_SOLICITATION_PHRASES = [
    "send me money", "send me ₹", "send me $",
    "pay me at", "pay to my upi", "pay to my account",
    "donate to my account", "donate to my upi",
    "transfer to my account",
    "use my referral code", "use my code", "sign up with my referral",
    "referral code:", "promo code:", "use code to earn",
    "click here to win", "you have won a prize",
    "congratulations you won", "claim your prize now",
    "free money guaranteed", "earn money fast", "get rich quick",
    "double your money", "guaranteed returns",
    # Task / review scams
    "review task", "review job", "earning task", "task earning",
    "earn per review", "earn per task",
    "telegram us for work", "whatsapp us for work",
    "dm for earning", "dm for work from home",
    "daily earning guarantee", "earn ₹ daily",
    "invitation code", "invite code",
    # Telegram/WhatsApp contact patterns (language-independent spam signal)
    "telegram id", "telegram per", "telegram par",
    "whatsapp id", "whatsapp per", "whatsapp par",
    "contact on telegram", "contact on whatsapp",
    # Review/task scam patterns
    "review work", "review ka ₹", "review ke",
    "task work", "work from home",
    # Referral / earning scam patterns
    "referral link", "share your referral", "start earning",
    "join our official channel",
    # Scam-app dashboard keywords (no currency symbol needed)
    "total income", "team profit", "current bonus",
    "daily profit", "total earnings", "total reward",
]


def is_spam_text(text: str) -> bool:
    """
    Keyword-based spam check for OCR-extracted image text.
    Uses a layered approach to minimise false positives.
    """
    text_lower = text.lower()

    # Layer 1: Direct high-confidence transaction phrases
    if any(phrase in text_lower for phrase in SPAM_TRANSACTION_PHRASES):
        return True

    # Layer 2: Multiple currency amounts = obvious spam (e.g. earning lists)
    currency_matches = SPAM_CURRENCY_RE.findall(text)
    has_currency = len(currency_matches) > 0
    if len(currency_matches) >= 3:
        return True

    # Layer 3: Currency amount + strong money-action verb together
    if has_currency:
        if any(verb in text_lower for verb in SPAM_STRONG_MONEY_VERBS):
            return True
        # Payment platform + currency is also a strong signal
        if any(platform in text_lower for platform in SPAM_PLATFORM_NAMES):
            return True

    # Layer 4: Currency + messaging app = scam (works in any language)
    if has_currency:
        has_messaging = any(app in text_lower for app in ["telegram", "whatsapp", "signal"])
        if has_messaging:
            return True

    # Layer 5: High-confidence solicitation / scam phrases
    if any(phrase in text_lower for phrase in SPAM_SOLICITATION_PHRASES):
        return True

    return False


# ─────────────────────────────────────────────
#  Request / Response Models
# ─────────────────────────────────────────────

class ContentIDMixin(BaseModel):
    """Exactly one of post_id, comment_id, or message_id must be provided."""
    post_id: Optional[str] = None
    comment_id: Optional[str] = None
    message_id: Optional[str] = None

    @property
    def content_id(self) -> str:
        """Return whichever ID was provided."""
        return self.post_id or self.comment_id or self.message_id or ""

    @property
    def content_id_type(self) -> str:
        if self.post_id:
            return "post_id"
        if self.comment_id:
            return "comment_id"
        if self.message_id:
            return "message_id"
        return "unknown"

    @model_validator(mode="after")
    def check_exactly_one_id(self):
        ids = [v for v in (self.post_id, self.comment_id, self.message_id) if v]
        if len(ids) != 1:
            raise ValueError("Exactly one of post_id, comment_id, or message_id must be provided")
        return self


class ImageURLRequest(ContentIDMixin):
    image_url: str


class TextRequest(ContentIDMixin):
    text: str


class VideoURLRequest(ContentIDMixin):
    video_url: str


class SafetyData(BaseModel):
    post_id: Optional[str] = None
    comment_id: Optional[str] = None
    message_id: Optional[str] = None
    rating: str
    category: str


class SafetyResponse(BaseModel):
    success: bool
    data: SafetyData
    message: str


def resolve_content_id(
    post_id: Optional[str] = None,
    comment_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> dict:
    """Validate and return a dict with the provided ID field for SafetyData."""
    ids = {k: v for k, v in [("post_id", post_id), ("comment_id", comment_id), ("message_id", message_id)] if v}
    if len(ids) != 1:
        raise HTTPException(status_code=422, detail="Exactly one of post_id, comment_id, or message_id must be provided")
    return ids


def get_log_id(id_dict: dict) -> str:
    """Return the single ID value for logging purposes."""
    return next(iter(id_dict.values()))


# ─────────────────────────────────────────────
#  Core helpers
# ─────────────────────────────────────────────

MAX_IMAGE_SIZE = 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


def resize_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    if max(w, h) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def strip_thinking(text: str) -> str:
    """Remove Qwen3 chain-of-thought <think>...</think> blocks."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def parse_model_response(response_text: str) -> dict:
    """
    Parse Rating + Category from model output.
    Always strips <think> content first so we read only the final answer.
    Falls back to 'needs_review' on parse failure so nothing slips through silently.
    """
    result = {"rating": "needs_review", "category": "NA"}

    # ✅ FIX 1: Strip thinking content before parsing
    response_text = strip_thinking(response_text)

    for line in response_text.strip().split("\n"):
        line_lower = line.lower().strip()

        if line_lower.startswith("rating:"):
            value = line.split(":", 1)[1].strip().lower()
            if "unsafe" in value:
                result["rating"] = "unsafe"
            elif "safe" in value:
                result["rating"] = "safe"

        elif line_lower.startswith("category:"):
            value = line.split(":", 1)[1].strip()
            for cat in ["11", "10", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                if cat in value:
                    result["category"] = cat
                    break

    # ✅ FIX 2: Unknown / unparseable response → flag for human review
    if result["rating"] == "needs_review":
        print(f"[WARN] Could not parse model response — flagging for review: {response_text!r}")

    return result


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_model(messages: list, max_tokens: int = 64) -> str:
    """
    Synchronous model inference. Always called inside model_lock via
    run_model_async so concurrent requests don't corrupt GPU state.
    """
    clear_memory()

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            # ✅ FIX 3: removed invalid past_key_values=None argument
        )

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del inputs, generated_ids, trimmed
    clear_memory()
    return raw


async def run_model_async(messages: list, max_tokens: int = 64) -> str:
    """
    Async wrapper that serialises all model calls through model_lock.
    Runs blocking inference in the default thread-pool executor.
    """
    # ✅ FIX 4: global lock prevents concurrent GPU corruption / OOM
    async with model_lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: run_model(messages, max_tokens)
        )


# ─────────────────────────────────────────────
#  Analysis functions
# ─────────────────────────────────────────────

async def analyze_image(image: Image.Image) -> dict:
    """
    Two-pass analysis:
      Pass 1 — visual safety classification via model
      Pass 2 — OCR text extraction + keyword-based spam check
    """
    try:
        image = resize_image(image)

        # ── Pass 1: Visual classification ──────────────────────────────
        raw = await run_model_async(
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": SAFETY_PROMPT},
                ],
            }],
            max_tokens=64,
        )
        print(f"[Image] Visual response: {raw!r}")
        visual_result = parse_model_response(raw)

        # Return immediately if visually unsafe or parse failed
        if visual_result["rating"] in ("unsafe", "needs_review"):
            return visual_result

        # ── Pass 2: OCR + keyword spam check ───────────────────────────
        ocr_raw = await run_model_async(
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "List every word or sentence of text visible in this image. "
                            "Output only the raw text, nothing else. "
                            "If there is no readable text, output exactly: NONE"
                        ),
                    },
                ],
            }],
            max_tokens=256,
        )
        extracted_text = strip_thinking(ocr_raw)
        print(f"[Image] Extracted text: {extracted_text!r}")

        if extracted_text and extracted_text.upper() != "NONE" and len(extracted_text) > 5:
            if is_spam_text(extracted_text):
                print("[Image] Spam detected via keyword match on OCR text")
                return {"rating": "unsafe", "category": "11"}

        return visual_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis error: {e}")


async def analyze_text(text: str) -> dict:
    """
    Text-only analysis — no dummy image.
    Keyword pre-check runs first to skip the model for obvious spam.
    """
    try:
        # Fast keyword pre-check before hitting the model
        if is_spam_text(text):
            print("[Text] Spam detected via keyword pre-check")
            return {"rating": "unsafe", "category": "11"}

        prompt = TEXT_SAFETY_PROMPT.format(text=text)

        # ✅ FIX 5: text-only message — no blank image
        raw = await run_model_async(
            [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }],
            max_tokens=64,
        )
        print(f"[Text] Model response: {raw!r}")
        return parse_model_response(raw)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis error: {e}")


# ─────────────────────────────────────────────
#  Image / video utilities
# ─────────────────────────────────────────────

def load_image_from_url(url: str) -> Image.Image:
    """Download image from URL with size guard."""
    try:
        with requests.get(url, timeout=30, stream=True) as resp:
            resp.raise_for_status()

            # ✅ FIX 6: reject oversized files before downloading fully
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=400, detail="Image exceeds 10 MB limit")

            content = resp.content
            if len(content) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=400, detail="Image exceeds 10 MB limit")

        return Image.open(BytesIO(content)).convert("RGB")

    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")


def extract_video_frames(video_path: str, num_frames: int = 5) -> list[Image.Image]:
    frames: list[Image.Image] = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        raise ValueError("Unable to read video frames")

    indices = [int(total * i / num_frames) for i in range(num_frames)]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

def log_request(post_id: str, endpoint: str, request_data: dict, response_data: dict):
    try:
        post_log_dir = os.path.join(LOGS_DIR, post_id)
        os.makedirs(post_log_dir, exist_ok=True)
        log_file = os.path.join(post_log_dir, "log.txt")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "request": request_data,
            "response": response_data,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry, indent=2))
            f.write("\n" + "=" * 80 + "\n")
    except Exception as e:
        print(f"[WARN] Logging error: {e}")


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "active", "model": MODEL_NAME}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_NAME}


# ── Image endpoints ────────────────────────────────────────────────────────────

ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/gif", "image/webp", "image/avif",
}


@app.post("/classify/image", response_model=SafetyResponse)
async def classify_image(
    file: UploadFile = File(...),
    post_id: Optional[str] = Form(None),
    comment_id: Optional[str] = Form(None),
    message_id: Optional[str] = Form(None),
):
    """Classify an uploaded image for safety."""
    id_dict = resolve_content_id(post_id, comment_id, message_id)

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    result = await analyze_image(image)

    response = SafetyResponse(
        success=True,
        data=SafetyData(**id_dict, **result),
        message="success",
    )
    log_request(get_log_id(id_dict), "/classify/image",
                {"file_name": file.filename, "content_type": file.content_type},
                response.dict())
    return response


@app.post("/classify/image/url", response_model=SafetyResponse)
async def classify_image_url(request: ImageURLRequest):
    """Classify an image from a URL for safety."""
    image = load_image_from_url(request.image_url)
    result = await analyze_image(image)

    id_dict = {request.content_id_type: request.content_id}
    response = SafetyResponse(
        success=True,
        data=SafetyData(**id_dict, **result),
        message="success",
    )
    log_request(request.content_id, "/classify/image/url",
                {"image_url": request.image_url}, response.dict())
    return response


# ── Text endpoint ──────────────────────────────────────────────────────────────

@app.post("/classify/text", response_model=SafetyResponse)
async def classify_text_content(request: TextRequest):
    """Classify text content for safety."""
    result = await analyze_text(request.text)

    id_dict = {request.content_id_type: request.content_id}
    response = SafetyResponse(
        success=True,
        data=SafetyData(**id_dict, **result),
        message="success",
    )
    log_request(request.content_id, "/classify/text",
                {"text": request.text}, response.dict())
    return response


# ── Video endpoints ────────────────────────────────────────────────────────────

async def _classify_frames(id_dict: dict, frames: list[Image.Image], endpoint: str, log_req: dict) -> SafetyResponse:
    """Shared frame-analysis logic for both video endpoints."""
    log_id = get_log_id(id_dict)
    for frame in frames:
        result = await analyze_image(frame)
        if result["rating"] in ("unsafe", "needs_review"):
            response = SafetyResponse(
                success=True,
                data=SafetyData(**id_dict, **result),
                message="success",
            )
            log_request(log_id, endpoint, log_req, response.dict())
            return response

    # ✅ FIX 7: consistent category "10" (not "NA") when all frames are safe
    response = SafetyResponse(
        success=True,
        data=SafetyData(**id_dict, rating="safe", category="10"),
        message="success",
    )
    log_request(log_id, endpoint, log_req, response.dict())
    return response


@app.post("/classify/video", response_model=SafetyResponse)
async def classify_video(
    file: UploadFile = File(...),
    post_id: Optional[str] = Form(None),
    comment_id: Optional[str] = Form(None),
    message_id: Optional[str] = Form(None),
):
    """Classify an uploaded video for safety (analyses key frames)."""
    id_dict = resolve_content_id(post_id, comment_id, message_id)

    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files are supported.")

    temp_path: Optional[str] = None
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(contents)
            temp_path = tmp.name

        frames = extract_video_frames(temp_path, num_frames=5)
        if not frames:
            raise ValueError("No frames extracted from video")

        return await _classify_frames(
            id_dict, frames,
            "/classify/video",
            {"file_name": file.filename, "content_type": file.content_type},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video classification failed: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@app.post("/classify/video/url", response_model=SafetyResponse)
async def classify_video_url(request: VideoURLRequest):
    """Classify a video from URL for safety."""
    id_dict = {request.content_id_type: request.content_id}
    temp_path: Optional[str] = None
    try:
        resp = requests.get(request.video_url, timeout=60)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(resp.content)
            temp_path = tmp.name

        frames = extract_video_frames(temp_path, num_frames=5)
        if not frames:
            raise ValueError("No frames extracted from video")

        return await _classify_frames(
            id_dict, frames,
            "/classify/video/url",
            {"video_url": request.video_url},
        )

    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video classification failed: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


# ── Logs endpoint ──────────────────────────────────────────────────────────────

@app.get("/logs/{content_id}")
async def get_logs(content_id: str):
    """Retrieve logs for a specific post_id, comment_id, or message_id."""
    log_file = os.path.join(LOGS_DIR, content_id, "log.txt")
    if not os.path.exists(log_file):
        raise HTTPException(status_code=404, detail=f"No logs found for id: {content_id}")

    try:
        with open(log_file) as f:
            content = f.read()
        return JSONResponse(content={"content_id": content_id, "logs": content})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading logs: {e}")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)