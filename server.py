"""
server.py — FastAPI inference server

Endpoints:
  POST /analyze        — image + question → answer
  POST /ask            — text-only question → answer
  GET  /health         — health check

Run:
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from model import AgriMultimodalModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration (override via environment variables) ────────────────
CFG = {
    "base_model":       os.getenv("BASE_MODEL",       "mistralai/Mistral-7B-v0.1"),
    "lora_weights":     os.getenv("LORA_WEIGHTS",     "./checkpoints/agri_lora"),
    "projector_path":   os.getenv("PROJECTOR_PATH",   "./checkpoints/projector_stage2_best.pt"),
    "load_in_4bit":     os.getenv("LOAD_4BIT",        "true").lower() == "true",
    "device":           os.getenv("DEVICE",           "cuda"),
    "max_new_tokens":   int(os.getenv("MAX_NEW_TOKENS", "512")),
    "temperature":      float(os.getenv("TEMPERATURE",   "0.7")),
}

# ── App lifespan (load model once on startup) ──────────────────────────
model: Optional[AgriMultimodalModel] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading AgriMultimodalModel …")
    try:
        model = AgriMultimodalModel(
            llm_base_model=CFG["base_model"],
            lora_weights_path=CFG["lora_weights"] if os.path.exists(CFG["lora_weights"]) else None,
            projector_path=CFG["projector_path"] if os.path.exists(CFG["projector_path"]) else None,
            load_in_4bit=CFG["load_in_4bit"],
            device=CFG["device"],
        )
        logger.info("Model ready ✓")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    del model

app = FastAPI(
    title="Agricultural Multimodal Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    question: str
    max_new_tokens: int = 512
    temperature: float = 0.7


class AnalysisResponse(BaseModel):
    answer: str
    latency_ms: float
    modality: str


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    image: UploadFile = File(..., description="Crop/plant image"),
    question: str = Form(..., description="Your question about the image"),
    max_new_tokens: int = Form(512),
    temperature: float = Form(0.7),
):
    """
    Multimodal endpoint: accepts an image + question, returns a grounded answer.
    The image is encoded as 256 visual patch tokens prepended to the question.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")

    # Validate image
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected image file, got {image.content_type}")

    raw = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")

    t0 = time.time()
    try:
        answer = model.generate(
            image=pil_img,
            question=question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(500, f"Inference failed: {e}")

    latency = (time.time() - t0) * 1000
    return AnalysisResponse(answer=answer, latency_ms=round(latency, 1),
                             modality="image+text")


@app.post("/ask", response_model=AnalysisResponse)
async def ask(req: TextRequest):
    """
    Text-only endpoint: useful when no image is available.
    Uses the agriculture-fine-tuned LLM directly.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.time()
    try:
        answer = model.generate_text_only(
            question=req.question,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(500, f"Inference failed: {e}")

    latency = (time.time() - t0) * 1000
    return AnalysisResponse(answer=answer, latency_ms=round(latency, 1),
                             modality="text")


@app.post("/analyze_base64", response_model=AnalysisResponse)
async def analyze_base64(payload: dict):
    """
    Alternative endpoint: accepts base64-encoded image + question as JSON.
    Useful for frontend clients that prefer JSON over multipart.

    Body: {"image_b64": "...", "question": "...", "max_new_tokens": 512}
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")

    try:
        img_bytes = base64.b64decode(payload["image_b64"])
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid base64 image")

    question = payload.get("question", "What disease is visible in this crop?")
    max_new_tokens = payload.get("max_new_tokens", 512)
    temperature = payload.get("temperature", 0.7)

    t0 = time.time()
    answer = model.generate(pil_img, question,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature)
    latency = (time.time() - t0) * 1000
    return AnalysisResponse(answer=answer, latency_ms=round(latency, 1),
                             modality="image+text")
