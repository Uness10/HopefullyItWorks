"""
model.py — Multimodal Agriculture Assistant
Vision encoder + MLP projector + fine-tuned LLM (LoRA)
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPVisionModel,
    CLIPImageProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from PIL import Image
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Patch Projector
# ─────────────────────────────────────────────

class PatchProjector(nn.Module):
    """
    Two-layer MLP that projects ViT patch embeddings into the LLM's
    embedding space so they can be concatenated with text token embeddings.

    vit_dim  : output dim of CLIP-ViT-Large  → 1024
    llm_dim  : hidden dim of Mistral-7B      → 4096
    """

    def __init__(self, vit_dim: int = 1024, llm_dim: int = 4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vit_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens : (B, N_patches, vit_dim)
        return self.mlp(patch_tokens)   # → (B, N_patches, llm_dim)


# ─────────────────────────────────────────────
#  Full Multimodal Model
# ─────────────────────────────────────────────

class AgriMultimodalModel(nn.Module):
    """
    Combines:
      - Frozen CLIP ViT (vision encoder)
      - Trained PatchProjector (alignment layer)
      - Agriculture-fine-tuned LLM with LoRA (reasoning engine)

    Image tokens are prepended to text tokens in the embedding space,
    giving the LLM full cross-attention over all visual patches.
    """

    # Special token used to separate image tokens from text in the sequence
    IMG_TOKEN = "<image>"

    def __init__(
        self,
        llm_base_model: str,
        lora_weights_path: Optional[str],
        projector_path: Optional[str],
        clip_model: str = "openai/clip-vit-large-patch14",
        load_in_4bit: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # ── 1. Vision encoder (always frozen) ──────────────────────────
        logger.info("Loading CLIP vision encoder …")
        self.vit = CLIPVisionModel.from_pretrained(clip_model)
        self.vit_processor = CLIPImageProcessor.from_pretrained(clip_model)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        # ── 2. LLM ─────────────────────────────────────────────────────
        logger.info("Loading base LLM …")
        quant_cfg = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if load_in_4bit
            else None
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_base_model,
            quantization_config=quant_cfg,
            torch_dtype=torch.float16 if not load_in_4bit else None,
            device_map="auto",
        )

        # Agriculture LoRA adapter
        if lora_weights_path:
            logger.info(f"Loading LoRA weights from {lora_weights_path} …")
            self.llm = PeftModel.from_pretrained(self.llm, lora_weights_path)

        self.tokenizer = AutoTokenizer.from_pretrained(llm_base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── 3. Projector ────────────────────────────────────────────────
        llm_hidden = self.llm.config.hidden_size          # e.g. 4096 for 7B
        vit_hidden = self.vit.config.hidden_size          # 1024 for ViT-Large
        self.projector = PatchProjector(vit_hidden, llm_hidden)

        if projector_path:
            logger.info(f"Loading projector weights from {projector_path} …")
            state = torch.load(projector_path, map_location="cpu")
            self.projector.load_state_dict(state)

        self.projector = self.projector.to(device).half()

    # ── Helpers ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Return projected patch tokens  →  (1, N_patches, llm_dim)"""
        inputs = self.vit_processor(images=image, return_tensors="pt").to(self.device)
        vit_out = self.vit(**inputs)
        # Drop the [CLS] token, keep all spatial patches
        patch_tokens = vit_out.last_hidden_state[:, 1:, :]   # (1, 256, 1024)
        projector_dtype = next(self.projector.parameters()).dtype
        patch_tokens = patch_tokens.to(dtype=projector_dtype)
        visual_embeds = self.projector(patch_tokens)          # (1, 256, 4096)
        return visual_embeds

    def _build_prompt(self, question: str) -> str:
        """Wrap the user question in a Mistral instruction template."""
        return (
            f"[INST] You are an expert agricultural assistant specialized in crop "
            f"diseases, pest management, and plant health. "
            f"An image of a crop/plant has been provided. "
            f"Answer the following question based on both the image and your knowledge.\n\n"
            f"Question: {question} [/INST]"
        )

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Full multimodal forward pass.

        Sequence fed to LLM:
          [visual patch tokens (256)] + [text prompt tokens]

        The LLM's self-attention can then attend across both.
        """
        # 1. Visual tokens
        visual_embeds = self._encode_image(image)          # (1, 256, D)

        # 2. Text embeddings
        prompt = self._build_prompt(question)
        text_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(self.device)

        embed_fn = self.llm.get_input_embeddings()
        text_embeds = embed_fn(text_ids).half()             # (1, T, D)

        # 3. Concatenate: [visual | text]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # (1, 256+T, D)

        # 4. Attention mask (attend to everything)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=self.device
        )

        # 5. Generate
        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 6. Decode only the newly generated tokens
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Strip the echoed prompt if present
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1].strip()

        return answer

    # ── Text-only fallback ────────────────────────────────────────────────

    @torch.inference_mode()
    def generate_text_only(
        self,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        prompt = self._build_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            repetition_penalty=1.1,
        )
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1].strip()
        return answer
