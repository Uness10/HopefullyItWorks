"""
inference.py — Run image+question inference with the full multimodal pipeline.

Loads:
  1. CLIP ViT (frozen visual encoder)
  2. PatchProjector (trained in Stage 1 / Stage 2)
  3. Qwen2.5-0.5B-Instruct + agriculture LoRA + visual LoRA (stacked)

Usage:
  # Single image + question
  python inference.py \
    --image /path/to/leaf.jpg \
    --question "What disease is visible on this leaf?" \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --agri_lora_ckpt /content/outputs/qwen_agri_lora \
    --visual_lora_ckpt /content/checkpoints/visual_lora_best \
    --projector_ckpt /content/checkpoints/projector_stage2_best.pt

  # Interactive REPL (loops until you type 'exit')
  python inference.py \
    --image /path/to/leaf.jpg \
    --interactive \
    ...same flags...
"""

import argparse
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPVisionModel,
    CLIPImageProcessor,
)
from peft import PeftModel

from model import PatchProjector


# ─────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────

def load_pipeline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Using device: {device}")

    # 1. CLIP ViT — frozen visual encoder
    print("[info] Loading CLIP ViT...")
    vit = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    vit_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False

    # 2. Qwen base LLM
    print(f"[info] Loading base LLM: {args.base_model}")
    llm = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="auto",
    )

    # 3. Stack LoRA adapters: agriculture LoRA first, then visual LoRA
    if args.agri_lora_ckpt:
        print(f"[info] Loading agriculture LoRA from: {args.agri_lora_ckpt}")
        llm = PeftModel.from_pretrained(
            llm,
            args.agri_lora_ckpt,
            adapter_name="agri",
        )

    if args.visual_lora_ckpt:
        print(f"[info] Loading visual LoRA from: {args.visual_lora_ckpt}")
        llm.load_adapter(args.visual_lora_ckpt, adapter_name="visual")
        # Merge both adapters so both are active during inference
        llm.set_adapter(["agri", "visual"])

    llm.eval()

    # 4. Projector
    print(f"[info] Loading projector from: {args.projector_ckpt}")
    projector = PatchProjector(
        vit_dim=vit.config.hidden_size,
        llm_dim=llm.config.hidden_size,
    ).to(device).half()
    projector.load_state_dict(
        torch.load(args.projector_ckpt, map_location="cpu")
    )
    projector.eval()

    # 5. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[info] Pipeline ready.\n")
    return vit, vit_proc, llm, projector, tokenizer, device


# ─────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────

def build_inputs(image_path, question, vit, vit_proc, llm, projector, tokenizer, device,
                 max_new_tokens=256):
    """
    Encode image + question into a single inputs_embeds tensor
    ready to feed into llm.generate().
    """
    # --- Visual encoding ---
    image = Image.open(image_path).convert("RGB")
    pixel_values = vit_proc(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        vit_out = vit(pixel_values=pixel_values)
    patch_tokens = vit_out.last_hidden_state[:, 1:, :]          # (1, 256, vit_dim)
    patch_tokens = patch_tokens.to(dtype=next(projector.parameters()).dtype)

    with torch.no_grad():
        visual_embeds = projector(patch_tokens)                  # (1, 256, llm_dim)

    # --- Text encoding ---
    prompt = (
        f"[INST] You are an expert agricultural assistant. "
        f"Analyze the crop image and answer: {question} [/INST] "
    )
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    text_embeds = llm.get_input_embeddings()(enc.input_ids)      # (1, T, llm_dim)

    # Align dtypes
    if visual_embeds.dtype != text_embeds.dtype:
        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)

    # Concatenate: [visual tokens | text tokens]
    combined = torch.cat([visual_embeds, text_embeds], dim=1)    # (1, 256+T, D)

    # Attention mask: attend to all visual + text tokens
    vis_mask = torch.ones(1, 256, dtype=torch.long, device=device)
    full_mask = torch.cat([vis_mask, enc.attention_mask], dim=1)

    return combined, full_mask


@torch.no_grad()
def answer(image_path, question, vit, vit_proc, llm, projector, tokenizer, device,
           max_new_tokens=256, temperature=0.7, do_sample=True):
    combined, full_mask = build_inputs(
        image_path, question,
        vit, vit_proc, llm, projector, tokenizer, device
    )

    output_ids = llm.generate(
        inputs_embeds=combined,
        attention_mask=full_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.strip()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to crop image")
    p.add_argument("--question", default=None, help="Question about the image")
    p.add_argument("--interactive", action="store_true",
                   help="Launch interactive Q&A loop for the given image")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--agri_lora_ckpt", default=None)
    p.add_argument("--visual_lora_ckpt", default=None)
    p.add_argument("--projector_ckpt", required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    vit, vit_proc, llm, projector, tokenizer, device = load_pipeline(args)

    if args.interactive:
        print(f"Image: {args.image}")
        print("Type your question (or 'exit' to quit):\n")
        while True:
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q:
                continue
            resp = answer(
                args.image, q,
                vit, vit_proc, llm, projector, tokenizer, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            print(f"Model: {resp}\n")
    else:
        if not args.question:
            raise ValueError("Provide --question or use --interactive mode.")
        resp = answer(
            args.image, args.question,
            vit, vit_proc, llm, projector, tokenizer, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\nAnswer: {resp}")