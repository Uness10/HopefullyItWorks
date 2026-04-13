from __future__ import annotations

import json
import os

import requests
import streamlit as st

st.set_page_config(page_title="Agri MLLM Chatbot", page_icon="seedling", layout="wide")

api_url = os.getenv("CHAT_API_URL", "http://127.0.0.1:8000/chat")

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Agri MLLM Chatbot")
st.caption("Text + image + PDF + optional audio (transcribed with Groq Whisper)")

with st.sidebar:
    st.subheader("Connection")
    st.code(api_url)
    if st.button("Clear Chat"):
        st.session_state.history = []

for turn in st.session_state.history:
    role = turn.get("role", "assistant")
    with st.chat_message(role):
        st.markdown(turn.get("content", ""))

text = st.chat_input("Ask an agriculture question...")

col1, col2, col3 = st.columns(3)
with col1:
    image_file = st.file_uploader("Attach image", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=False)
with col2:
    pdf_file = st.file_uploader("Attach PDF", type=["pdf"], accept_multiple_files=False)
with col3:
    audio_file = st.file_uploader("Attach audio", type=["wav", "mp3", "flac", "m4a", "ogg"], accept_multiple_files=False)

if text:
    with st.chat_message("user"):
        st.markdown(text)

    st.session_state.history.append({"role": "user", "content": text})

    form_data = {
        "message": text,
        "history_json": json.dumps(st.session_state.history[-10:]),
    }

    files = {}
    if image_file is not None:
        files["image"] = (image_file.name, image_file.getvalue(), image_file.type or "application/octet-stream")
    if pdf_file is not None:
        files["pdf"] = (pdf_file.name, pdf_file.getvalue(), pdf_file.type or "application/pdf")
    if audio_file is not None:
        files["audio"] = (audio_file.name, audio_file.getvalue(), audio_file.type or "application/octet-stream")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(api_url, data=form_data, files=files, timeout=300)
                response.raise_for_status()
                payload = response.json()
                answer = payload.get("answer", "")
                transcript = payload.get("transcript")
                used_modalities = payload.get("used_modalities", [])

                if transcript:
                    st.caption("Audio transcript")
                    st.code(transcript)

                if used_modalities:
                    st.caption("Used modalities: " + ", ".join(used_modalities))

                st.markdown(answer)
                st.session_state.history.append({"role": "assistant", "content": answer})
            except Exception as exc:  # noqa: BLE001
                error_message = f"Request failed: {exc}"
                st.error(error_message)
                st.session_state.history.append({"role": "assistant", "content": error_message})
