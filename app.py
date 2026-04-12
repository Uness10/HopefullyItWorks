"""
app.py — Streamlit frontend for the Agricultural Multimodal Assistant

Run:
  streamlit run app.py
"""

import io
import base64
import time
import requests
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroVision — Crop Disease Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0d1f0f 0%, #132815 50%, #0a1a0b 100%);
    color: #e8f5e2;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 35, 16, 0.95);
    border-right: 1px solid rgba(74, 155, 64, 0.2);
}

/* Cards */
.analysis-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(74,155,64,0.25);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
}

/* Answer block */
.answer-block {
    background: rgba(74,155,64,0.08);
    border-left: 3px solid #4a9b40;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    font-size: 1.05rem;
    line-height: 1.75;
    color: #d4eecf;
}

/* Metric chips */
.metric-chip {
    display: inline-block;
    background: rgba(74,155,64,0.15);
    border: 1px solid rgba(74,155,64,0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    color: #8dc888;
    margin-right: 6px;
}

/* Suggested questions */
.suggestion-btn {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(74,155,64,0.2) !important;
    color: #b5d9b0 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3a8a30, #4fa844) !important;
    border: none !important;
    color: white !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.8rem !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(74,155,64,0.4) !important;
}

/* Input fields */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(74,155,64,0.25) !important;
    color: #e8f5e2 !important;
    border-radius: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #8dc888 !important;
}
.stTabs [aria-selected="true"] {
    color: #4a9b40 !important;
    border-bottom-color: #4a9b40 !important;
}

/* File uploader */
.stFileUploader {
    border: 2px dashed rgba(74,155,64,0.3) !important;
    border-radius: 12px !important;
    background: rgba(74,155,64,0.04) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(74,155,64,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

SUGGESTED_QUESTIONS = [
    "What disease symptoms are visible on this plant?",
    "Is this leaf showing signs of fungal infection?",
    "What treatment would you recommend for this condition?",
    "How severe is the damage visible in this image?",
    "What nutrient deficiency might cause these symptoms?",
    "Compare these symptoms to common tomato diseases.",
]

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 AgroVision")
    st.markdown("*Agricultural AI Assistant*")
    st.divider()

    st.markdown("### ⚙️ Generation Settings")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05,
                            help="Higher = more creative answers")
    max_tokens = st.slider("Max tokens", 128, 1024, 512, 64)

    st.divider()
    st.markdown("### 📋 About")
    st.markdown("""
    This assistant combines:
    - **CLIP ViT** vision encoder
    - **MLP projector** (256 patch tokens)
    - **Agriculture-finetuned LLM**

    Upload a crop image and ask any plant health question.
    """)

    st.divider()

    # Server health check
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.json().get("model_loaded"):
            st.success("🟢 Server online")
        else:
            st.warning("🟡 Server loading…")
    except Exception:
        st.error("🔴 Server offline\n\nStart with:\n```\nuvicorn server:app\n```")


# ── Header ─────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("<div style='font-size:3rem;padding-top:8px'>🌾</div>",
                unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <h1 style='margin:0;padding-top:4px;color:#8dc888;font-size:2.2rem'>
        AgroVision
    </h1>
    <p style='margin:0;color:#5a8a55;font-size:1rem'>
        Crop Disease & Plant Health Multimodal Assistant
    </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Main tabs ──────────────────────────────────────────────────────────
tab_image, tab_text, tab_history = st.tabs(
    ["🖼️  Image Analysis", "💬  Text Q&A", "📜  History"]
)

# ── Session state ─────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "prefill_question" not in st.session_state:
    st.session_state.prefill_question = ""


# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — Image Analysis
# ══════════════════════════════════════════════════════════════════════
with tab_image:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Upload Crop Image")
        uploaded_file = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded image", use_column_width=True)

            # Image metadata chips
            w, h = img.size
            mode = img.mode
            size_kb = len(uploaded_file.getvalue()) // 1024
            st.markdown(
                f'<span class="metric-chip">{w}×{h}px</span>'
                f'<span class="metric-chip">{mode}</span>'
                f'<span class="metric-chip">{size_kb} KB</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### Suggested Questions")
        for q in SUGGESTED_QUESTIONS:
            if st.button(q, key=f"sq_{q[:20]}", use_container_width=True):
                st.session_state.prefill_question = q
                st.rerun()

    with right:
        st.markdown("### Your Question")

        question = st.text_area(
            "Question",
            value=st.session_state.prefill_question,
            placeholder="What disease symptoms are visible on this leaf?",
            height=100,
            label_visibility="collapsed",
        )

        # Reset prefill after use
        if st.session_state.prefill_question:
            st.session_state.prefill_question = ""

        analyze_btn = st.button(
            "🔍 Analyze Image",
            type="primary",
            disabled=not uploaded_file or not question.strip(),
            use_container_width=True,
        )

        if analyze_btn and uploaded_file and question.strip():
            with st.spinner("Processing image through vision encoder…"):
                try:
                    # Re-read file bytes
                    uploaded_file.seek(0)
                    raw_bytes = uploaded_file.read()

                    resp = requests.post(
                        f"{API_BASE}/analyze",
                        files={"image": (uploaded_file.name, raw_bytes,
                                         uploaded_file.type)},
                        data={
                            "question": question,
                            "max_new_tokens": max_tokens,
                            "temperature": temperature,
                        },
                        timeout=120,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        latency = data["latency_ms"]

                        st.markdown(
                            f'<div class="analysis-card">'
                            f'<p style="color:#5a8a55;font-size:0.8rem;margin-bottom:0.5rem">'
                            f'🌿 Analysis Result &nbsp;'
                            f'<span class="metric-chip">{latency:.0f} ms</span>'
                            f'<span class="metric-chip">256 patch tokens</span>'
                            f'</p>'
                            f'<div class="answer-block">{answer}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Save to history
                        thumb_b64 = _img_to_b64_thumb(img)
                        st.session_state.history.insert(0, {
                            "type": "image",
                            "question": question,
                            "answer": answer,
                            "latency_ms": latency,
                            "thumb": thumb_b64,
                        })
                    else:
                        st.error(f"Server error {resp.status_code}: {resp.text}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to server. Is it running?")
                except Exception as e:
                    st.error(f"Error: {e}")

        elif not uploaded_file:
            st.info("👆 Upload an image to get started")
        elif not question.strip():
            st.info("✏️ Enter a question about your crop")


# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — Text-only Q&A
# ══════════════════════════════════════════════════════════════════════
with tab_text:
    st.markdown("### Ask Without an Image")
    st.markdown(
        "<p style='color:#5a8a55;font-size:0.9rem'>"
        "Use the agriculture-finetuned LLM directly for knowledge questions."
        "</p>",
        unsafe_allow_html=True,
    )

    text_q = st.text_area(
        "Question",
        placeholder="What are the early symptoms of late blight in tomatoes?",
        height=120,
        label_visibility="collapsed",
    )

    ask_btn = st.button("💬 Ask", type="primary",
                        disabled=not text_q.strip())

    if ask_btn and text_q.strip():
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/ask",
                    json={
                        "question": text_q,
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(
                        f'<div class="analysis-card">'
                        f'<p style="color:#5a8a55;font-size:0.8rem;margin-bottom:0.5rem">'
                        f'💬 Answer &nbsp;<span class="metric-chip">{data["latency_ms"]:.0f} ms</span>'
                        f'<span class="metric-chip">text only</span></p>'
                        f'<div class="answer-block">{data["answer"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.history.insert(0, {
                        "type": "text",
                        "question": text_q,
                        "answer": data["answer"],
                        "latency_ms": data["latency_ms"],
                    })
                else:
                    st.error(f"Server error: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to server.")


# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — History
# ══════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### Conversation History")

    if not st.session_state.history:
        st.markdown(
            "<p style='color:#5a8a55'>No queries yet. Start by analyzing an image.</p>",
            unsafe_allow_html=True,
        )
    else:
        if st.button("🗑 Clear history"):
            st.session_state.history = []
            st.rerun()

        for i, entry in enumerate(st.session_state.history):
            icon = "🖼️" if entry["type"] == "image" else "💬"
            with st.expander(
                f"{icon}  {entry['question'][:80]}…"
                if len(entry["question"]) > 80
                else f"{icon}  {entry['question']}"
            ):
                if entry["type"] == "image" and "thumb" in entry:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        img_data = base64.b64decode(entry["thumb"])
                        st.image(Image.open(io.BytesIO(img_data)),
                                 use_column_width=True)
                    with col2:
                        st.markdown(
                            f'<div class="answer-block">{entry["answer"]}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        f'<div class="answer-block">{entry["answer"]}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<span class="metric-chip">{entry["latency_ms"]:.0f} ms</span>'
                    f'<span class="metric-chip">{entry["type"]}</span>',
                    unsafe_allow_html=True,
                )


# ── Helper functions ───────────────────────────────────────────────────
def _img_to_b64_thumb(img: Image.Image, size=(120, 120)) -> str:
    """Resize image and encode as base64 for thumbnail storage."""
    thumb = img.copy()
    thumb.thumbnail(size)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()
