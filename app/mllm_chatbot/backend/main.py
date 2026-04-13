from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from .inference import AgriChatEngine


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _parse_history(history_json: str | None) -> list[dict[str, str]]:
    if not history_json:
        return []
    try:
        value = json.loads(history_json)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []
    except json.JSONDecodeError:
        return []


app = FastAPI(title="Agri MLLM Chat API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = AgriChatEngine(
    model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
    adapter_dir=os.getenv("ADAPTER_DIR", ""),
    device=os.getenv("DEVICE", "cuda"),
    max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "160")),
)


@app.get("/", include_in_schema=False, response_model=None)
def home():
    if FRONTEND_HTML.exists():
        return FileResponse(str(FRONTEND_HTML), media_type="text/html")
    return HTMLResponse("<h3>Frontend file not found.</h3>", status_code=404)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(
    message: Annotated[str, Form(...)],
    history_json: Annotated[str | None, Form()] = None,
    image: Annotated[UploadFile | None, File()] = None,
    pdf: Annotated[UploadFile | None, File()] = None,
    audio: Annotated[UploadFile | None, File()] = None,
) -> dict[str, object]:
    try:
        image_bytes = await image.read() if image is not None else None
        pdf_bytes = await pdf.read() if pdf is not None else None
        audio_bytes = await audio.read() if audio is not None else None
        audio_filename = audio.filename if audio is not None and audio.filename else "audio.wav"

        history = _parse_history(history_json)
        result = engine.chat(
            message=message,
            history=history,
            image_bytes=image_bytes,
            pdf_bytes=pdf_bytes,
            audio_bytes=audio_bytes,
            audio_filename=audio_filename,
        )
        return {
            "answer": result.answer,
            "transcript": result.transcript,
            "used_modalities": result.used_modalities,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
