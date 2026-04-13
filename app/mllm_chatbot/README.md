# MLLM Chatbot App

This folder contains a separate chatbot app with:
- FastAPI backend for multimodal inference
- Custom HTML/CSS/JS frontend chat UI

## Features

- Text chat with the base LLM
- Optional image upload (uses image adapter if `ADAPTER_DIR` is set)
- Optional PDF upload (extracts figure images and uses PDF adapter if available)
- Optional audio upload (transcribed with Groq Whisper, then fed as text)

## Directory

- `backend/main.py`: FastAPI API server
- `backend/inference.py`: inference engine
- `frontend/index.html`: custom UI served by FastAPI (`/`)
- `frontend/app.py`: legacy Streamlit UI
- `requirements.txt`: app dependencies
- `.env.example`: environment variable template

## Quick Start

1. Install dependencies:

```bash
pip install -r apps/mllm_chatbot/requirements.txt
pip install -e .
```

2. Export env vars (adapt paths/keys):

```bash
export MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
export ADAPTER_DIR="outputs/qwen_agri_lora"
export DEVICE="cuda"
export GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

3. Start backend:

```bash
uvicorn apps.mllm_chatbot.backend.main:app --host 0.0.0.0 --port 8000
```

4. Open the UI:

```text
http://127.0.0.1:8000/
```

Optional legacy Streamlit frontend (new terminal):

```bash
streamlit run apps/mllm_chatbot/frontend/app.py
```

## Notes

- If `ADAPTER_DIR` points to a PEFT LoRA folder (for example `outputs/qwen_agri_lora`), backend loads the fine-tuned text adapter.
- If `ADAPTER_DIR` points to a multimodal adapter folder containing `adapter_state.pt`, backend loads multimodal projectors.
- If `ADAPTER_DIR` is empty or invalid, backend runs base text-only LLM mode.
- If audio is uploaded, backend calls Groq Whisper transcription API and injects transcript into the text prompt.
- PDF inference uses first extracted figures from the uploaded PDF.
