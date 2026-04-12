"""
train_projector.py — Two-stage training for the visual alignment layer

Stage 1 : Train PatchProjector only  (LLM + ViT frozen)
          Data: (image, caption) pairs
          Loss: causal LM on caption tokens

Stage 2 : Train PatchProjector + LoRA in LLM  (ViT frozen)
          Data: (image, question, answer) VQA triplets
          Loss: causal LM on answer tokens only

Run:
  # Stage 1
  python train_projector.py --stage 1 --data_dir ./data/captions

  # Stage 2
  python train_projector.py --stage 2 --data_dir ./data/vqa \
      --projector_ckpt ./checkpoints/projector_stage1.pt \
      --lora_ckpt ./checkpoints/agri_lora
"""

import os
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPVisionModel,
    CLIPImageProcessor,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from tqdm import tqdm

from model import PatchProjector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Datasets
# ─────────────────────────────────────────────

class CaptionDataset(Dataset):
    """
    Stage 1 dataset.
    Expects a JSONL file where each line is:
        {"image": "path/to/img.jpg", "caption": "Leaf shows ..."}
    """
    def __init__(self, jsonl_path: str, vit_processor, tokenizer, max_len=128):
        self.samples = [json.loads(l) for l in open(jsonl_path)]
        self.vit_processor = vit_processor
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image"]).convert("RGB")
        pixel_values = self.vit_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        cap = self.tokenizer(
            s["caption"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": cap.input_ids.squeeze(0),
            "attention_mask": cap.attention_mask.squeeze(0),
        }


class VQADataset(Dataset):
    """
    Stage 2 dataset.
    Expects a JSONL file where each line is:
        {"image": "path/to/img.jpg",
         "question": "What disease is visible?",
         "answer": "The leaf shows early-stage ..."}
    """
    def __init__(self, jsonl_path: str, vit_processor, tokenizer,
                 max_q_len=64, max_a_len=256):
        self.samples = [json.loads(l) for l in open(jsonl_path)]
        self.vit_processor = vit_processor
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image"]).convert("RGB")
        pixel_values = self.vit_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        prompt = (
            f"[INST] You are an expert agricultural assistant. "
            f"Analyze the crop image and answer: {s['question']} [/INST] "
        )
        full_text = prompt + s["answer"] + self.tokenizer.eos_token

        enc = self.tokenizer(
            full_text,
            max_length=self.max_q_len + self.max_a_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Build labels: mask prompt tokens with -100 so loss is only on answer
        prompt_len = len(self.tokenizer(prompt).input_ids)
        labels = enc.input_ids.clone().squeeze(0)
        labels[:prompt_len] = -100
        labels[enc.attention_mask.squeeze(0) == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": labels,
        }


# ─────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────

def build_combined_embeds(
    pixel_values, input_ids, vit, projector, llm, device
):
    """
    Encode image + text and concatenate in embedding space.
    Returns (inputs_embeds, attention_mask_extended).
    """
    B = pixel_values.size(0)

    # Visual
    with torch.no_grad():
        vit_out = vit(pixel_values=pixel_values.to(device))
    patch_tokens = vit_out.last_hidden_state[:, 1:, :]   # (B, 256, vit_dim)
    projector_dtype = next(projector.parameters()).dtype
    patch_tokens = patch_tokens.to(dtype=projector_dtype)
    visual_embeds = projector(patch_tokens)               # (B, 256, llm_dim)

    # Text
    text_embeds = llm.get_input_embeddings()(
        input_ids.to(device)
    )  # (B, T, llm_dim)

    # Keep modalities in the same dtype before concatenation.
    if visual_embeds.dtype != text_embeds.dtype:
        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)

    combined = torch.cat([visual_embeds, text_embeds], dim=1)  # (B, 256+T, D)

    # Extend attention mask to cover visual tokens (always attend)
    vis_mask = torch.ones(B, 256, dtype=torch.long, device=device)
    return combined, vis_mask


def compute_loss(combined_embeds, vis_mask, input_ids, attention_mask,
                 labels, llm, device):
    full_mask = torch.cat(
        [vis_mask, attention_mask.to(device)], dim=1
    )

    # Shift labels: visual tokens have no ground-truth, use -100
    vis_labels = torch.full((labels.size(0), 256), -100,
                            dtype=torch.long, device=device)
    full_labels = torch.cat([vis_labels, labels.to(device)], dim=1)

    out = llm(
        inputs_embeds=combined_embeds,
        attention_mask=full_mask,
        labels=full_labels,
    )
    return out.loss


def get_llm_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


# ─────────────────────────────────────────────
#  Stage 1 — projector-only training
# ─────────────────────────────────────────────

def train_stage1(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Stage 1 training on {device}")

    vit = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    vit_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    for p in vit.parameters():
        p.requires_grad = False
    vit.eval()

    llm_dtype = get_llm_dtype(device)
    llm = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=llm_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if args.lora_ckpt:
        llm = PeftModel.from_pretrained(llm, args.lora_ckpt)
    for p in llm.parameters():
        p.requires_grad = False
    llm.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    projector = PatchProjector(
        vit_dim=vit.config.hidden_size,
        llm_dim=llm.config.hidden_size,
    ).to(device)

    captions_path = os.path.join(args.data_dir, "captions.jsonl")
    if not os.path.exists(captions_path):
        raise FileNotFoundError(
            "Stage 1 requires captions.jsonl at: "
            f"{captions_path}\n"
            "Generate it first with build_stage1_captions.py, then rerun training."
        )

    dataset = CaptionDataset(captions_path, vit_proc, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=2)

    optimizer = AdamW(projector.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(loader) * args.epochs)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        projector.train()
        total_loss = 0.0
        valid_steps = 0
        skipped_steps = 0

        for step, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"), start=1
        ):
            optimizer.zero_grad(set_to_none=True)

            combined, vis_mask = build_combined_embeds(
                batch["pixel_values"], batch["input_ids"],
                vit, projector, llm, device
            )

            # For stage 1, treat caption ids as both input and labels
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100

            loss = compute_loss(
                combined, vis_mask,
                batch["input_ids"], batch["attention_mask"],
                labels, llm, device
            )

            if not torch.isfinite(loss):
                skipped_steps += 1
                logger.warning(
                    f"Epoch {epoch+1} Batch {step}/{len(loader)} — non-finite loss; skipping batch."
                )
                continue

            logger.info(
                f"Epoch {epoch+1} Batch {step}/{len(loader)} — loss: {loss.item():.6f}"
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            valid_steps += 1

        avg = total_loss / max(valid_steps, 1)
        logger.info(
            f"Epoch {epoch+1} — loss: {avg:.4f} "
            f"(valid_steps={valid_steps}, skipped_steps={skipped_steps})"
        )

    ckpt_path = os.path.join(args.output_dir, "projector_stage1.pt")
    torch.save(projector.state_dict(), ckpt_path)
    logger.info(f"Projector saved to {ckpt_path}")


# ─────────────────────────────────────────────
#  Stage 2 — projector + visual LoRA training
# ─────────────────────────────────────────────

def train_stage2(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Stage 2 training on {device}")

    vit = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    vit_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    for p in vit.parameters():
        p.requires_grad = False
    vit.eval()

    # Load base LLM + existing agriculture LoRA
    llm_dtype = get_llm_dtype(device)
    llm = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=llm_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if args.lora_ckpt:
        llm = PeftModel.from_pretrained(llm, args.lora_ckpt)

    # Add a NEW LoRA adapter on top for visual reasoning
    visual_lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, visual_lora_cfg)
    llm.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load pre-trained projector from stage 1
    projector = PatchProjector(
        vit_dim=vit.config.hidden_size,
        llm_dim=llm.config.hidden_size,
    ).to(device)
    if args.projector_ckpt:
        if not os.path.exists(args.projector_ckpt):
            raise FileNotFoundError(
                "Stage 2 projector checkpoint not found: "
                f"{args.projector_ckpt}\n"
                "Run stage 1 first or fix --projector_ckpt path."
            )
        projector.load_state_dict(
            torch.load(args.projector_ckpt, map_location="cpu")
        )
        logger.info("Loaded stage-1 projector weights")
    else:
        raise ValueError(
            "Stage 2 requires --projector_ckpt (for example ./checkpoints/projector_stage1.pt)."
        )

    vqa_path = os.path.join(args.data_dir, "vqa.jsonl")
    if not os.path.exists(vqa_path):
        raise FileNotFoundError(
            "Stage 2 requires vqa.jsonl at: "
            f"{vqa_path}\n"
            "Generate it first with build_stage2_vqa.py, then rerun training."
        )

    dataset = VQADataset(vqa_path, vit_proc, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=2)

    trainable = list(projector.parameters()) + \
                [p for p in llm.parameters() if p.requires_grad]

    optimizer = AdamW(trainable, lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        projector.train()
        llm.train()
        total_loss = 0.0
        valid_steps = 0
        skipped_steps = 0

        for step, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"), start=1
        ):
            optimizer.zero_grad(set_to_none=True)

            combined, vis_mask = build_combined_embeds(
                batch["pixel_values"], batch["input_ids"],
                vit, projector, llm, device
            )

            loss = compute_loss(
                combined, vis_mask,
                batch["input_ids"], batch["attention_mask"],
                batch["labels"], llm, device
            )

            if not torch.isfinite(loss):
                skipped_steps += 1
                logger.warning(
                    f"Epoch {epoch+1} Batch {step}/{len(loader)} — non-finite loss; skipping batch."
                )
                continue

            logger.info(
                f"Epoch {epoch+1} Batch {step}/{len(loader)} — loss: {loss.item():.6f}"
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            valid_steps += 1

        avg = total_loss / max(valid_steps, 1)
        logger.info(
            f"Epoch {epoch+1} — loss: {avg:.4f} "
            f"(valid_steps={valid_steps}, skipped_steps={skipped_steps})"
        )

        if avg < best_loss:
            best_loss = avg
            proj_path = os.path.join(args.output_dir, "projector_stage2_best.pt")
            lora_path = os.path.join(args.output_dir, "visual_lora_best")
            torch.save(projector.state_dict(), proj_path)
            llm.save_pretrained(lora_path)
            logger.info(f"  ✓ Checkpoint saved (loss={avg:.4f})")

    logger.info("Stage 2 training complete.")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True, choices=[1, 2])
    p.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--lora_ckpt", default=None,
                   help="Path to agriculture LoRA weights")
    p.add_argument("--projector_ckpt", default=None,
                   help="Stage-1 projector checkpoint (required for stage 2)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()

    if args.stage == 1:
        train_stage1(args)
    else:
        train_stage2(args)
