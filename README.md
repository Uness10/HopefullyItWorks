# Multimodal Agriculture Assistant

This project builds a multimodal agriculture assistant that can:
- Answer text-only agronomy questions.
- Analyze crop leaf images and provide disease-oriented guidance.

The implementation follows a three-stage training pipeline using:
- Base LLM: `Qwen/Qwen2.5-0.5B-Instruct`
- Vision encoder: `openai/clip-vit-large-patch14`
- Parameter-efficient tuning with LoRA and a learned visual projector

The pipeline and results are documented in `docs/report.md`.

## Pipeline Summary

1. Stage 0: Domain text adaptation (LoRA)
     - Train LoRA on agriculture QA text.
     - Output: `outputs/qwen_agri_lora`

2. Stage 1: Vision-language alignment
     - Freeze ViT + LLM, train only projector on image-caption pairs.
     - Output: `checkpoints/projector_stage1.pt`

3. Stage 2: Multimodal refinement
     - Freeze ViT, train projector + visual LoRA on VQA triplets.
     - Output: `checkpoints/projector_stage2_best.pt` and `checkpoints/visual_lora_best`

## Report Highlights

Based on `docs/report.md`:
- Text LoRA training loss decreased from ~3.449 to ~0.642.
- Stage 1 projector average loss dropped from 0.5500 to 0.0090 (3 epochs).
- Stage 2 multimodal average loss dropped from 0.3996 to 0.0141 (3 epochs).
- Final inference on PlantVillage apple leaf images produced disease-aware diagnostic reasoning.

## Repository Structure

- `model/`: training scripts, inference script, notebook workflow
- `app/mllm_chatbot/`: FastAPI + UI app (optional serving layer)
- `docs/`: report, logs, and presentation material
- `requirements.txt`: main dependencies

## Prerequisites

- Python 3.10+ (recommended)
- Git
- CUDA-enabled GPU recommended for training
- Optional: Hugging Face token for faster dataset/model downloads (`HF_TOKEN`)

## Step-by-Step: Run the Project

The recommended path is the notebook workflow in `model/main.ipynb`.
Run all commands below from the repository root.

### 1) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Prepare folders

Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force -Path data/captions, data/vqa, checkpoints, outputs | Out-Null
```

macOS/Linux:

```bash
mkdir -p data/captions data/vqa checkpoints outputs
```

### 4) Get PlantVillage dataset

```bash
git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git data/PlantVillage-Dataset
```

Expected image root for next steps:
- `data/PlantVillage-Dataset/raw/color`

### 5) Run Stage 0 (text LoRA)

```bash
python -m model.fine_tune \
    --output-dir outputs/qwen_agri_lora \
    --dataset-output-jsonl data/processed/agri_train.jsonl
```

### 6) Build Stage 1 caption dataset

```bash
python -m model.build_stage1_captions \
    --images_dir data/PlantVillage-Dataset/raw/color \
    --output_jsonl data/captions/captions.jsonl \
    --shuffle \
    --max_images 1000
```

### 7) Build Stage 2 VQA dataset

```bash
python -m model.build_stage2_vqa \
    --images_dir data/PlantVillage-Dataset/raw/color \
    --output_jsonl data/vqa/vqa.jsonl \
    --pairs_per_image 4 \
    --shuffle \
    --max_images 250
```

### 8) Train Stage 1 projector

```bash
python -m model.train_projector \
    --stage 1 \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_ckpt outputs/qwen_agri_lora \
    --data_dir data/captions \
    --output_dir checkpoints \
    --epochs 3 \
    --batch_size 4 \
    --lr 1e-3
```

### 9) Train Stage 2 multimodal alignment

```bash
python -m model.train_projector \
    --stage 2 \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_ckpt outputs/qwen_agri_lora \
    --projector_ckpt checkpoints/projector_stage1.pt \
    --data_dir data/vqa \
    --output_dir checkpoints \
    --epochs 3 \
    --batch_size 4 \
    --lr 1e-3
```

### 10) Run final multimodal inference

```bash
python -m model.inference \
    --image "data/PlantVillage-Dataset/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG" \
    --question "What disease is visible and how severe is it?" \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --agri_lora_ckpt outputs/qwen_agri_lora \
    --visual_lora_ckpt checkpoints/visual_lora_best \
    --projector_ckpt checkpoints/projector_stage2_best.pt
```

## Notebook Workflow (Alternative)

You can run the complete pipeline directly from `model/main.ipynb`.

Important:
- Some cells use Colab-style absolute paths (`/content/...`).
- If running locally, replace those paths with repository-relative paths used above.

## Optional: Run the Chatbot App

There is an app under `app/mllm_chatbot/` with FastAPI + frontend.
The backend currently imports an `agri_mllm` package; make sure it is available in your environment before starting the API.

Install app dependencies:

```bash
pip install -r app/mllm_chatbot/requirements.txt
```

Start API server:

```bash
uvicorn app.mllm_chatbot.backend.main:app --host 0.0.0.0 --port 8000
```

Open:
- `http://127.0.0.1:8000/`

## Output Artifacts

- `outputs/qwen_agri_lora`: agriculture text LoRA adapter
- `checkpoints/projector_stage1.pt`: stage 1 projector checkpoint
- `checkpoints/projector_stage2_best.pt`: best stage 2 projector checkpoint
- `checkpoints/visual_lora_best`: best visual LoRA adapter

## Troubleshooting

- If Hugging Face downloads are slow/rate-limited, set `HF_TOKEN`.
- If CUDA OOM occurs, reduce `--batch_size` and/or `--max_images`.
- Keep stage checkpoints so you can resume without restarting from scratch.
- Ensure dataset path exists before stage data generation commands.