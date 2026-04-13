from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from agri_mllm.modeling.architecture import AgricultureMultiInputModel
from agri_mllm.modeling.config import AgricultureModelConfig, ProjectorConfig


@dataclass
class ChatResult:
    answer: str
    transcript: str | None
    used_modalities: list[str]


class AgriChatEngine:
    def __init__(
        self,
        model_name: str,
        adapter_dir: str | None,
        device: str,
        max_new_tokens: int,
    ) -> None:
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.multimodal_model: AgricultureMultiInputModel | None = None
        self.text_model: nn.Module | None = None

        self.vision_processor: AutoProcessor | None = None
        self.vision_model: AutoModel | None = None

        if adapter_dir:
            adapter_path = Path(adapter_dir)
            state_path = adapter_path / "adapter_state.pt"
            args_path = adapter_path / "training_args.json"
            peft_adapter_config = adapter_path / "adapter_config.json"
            peft_adapter_weights = adapter_path / "adapter_model.safetensors"

            if state_path.exists() and args_path.exists():
                self.multimodal_model = self._load_multimodal_model(adapter_path, model_name)
            elif peft_adapter_config.exists() and peft_adapter_weights.exists():
                self.text_model = self._load_text_lora_model(adapter_path, model_name)

        if self.multimodal_model is None and self.text_model is None:
            self.text_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()

    def _load_text_lora_model(self, adapter_dir: Path, base_model_name: str) -> nn.Module:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "Loading LoRA adapters requires 'peft'. Install it with: pip install peft>=0.12.0"
            ) from exc

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        return model.to(self.device).eval()

    def _load_multimodal_model(self, adapter_dir: Path, fallback_model_name: str) -> AgricultureMultiInputModel:
        state = torch.load(adapter_dir / "adapter_state.pt", map_location="cpu")
        train_args = json.loads((adapter_dir / "training_args.json").read_text(encoding="utf-8"))

        model_name = train_args.get("model_name", fallback_model_name)
        hidden = AutoConfig.from_pretrained(model_name).hidden_size

        def in_dim(prefix: str) -> int:
            key = f"{prefix}.network.0.weight"
            if key not in state:
                raise KeyError(f"Missing '{key}' in adapter checkpoint.")
            return int(state[key].shape[1])

        cfg = AgricultureModelConfig(
            llm_hidden_size=hidden,
            crop_image=ProjectorConfig(
                input_dim=in_dim("crop_image_projector"),
                hidden_dim=int(train_args.get("projector_hidden_dim", 1024)),
                output_dim=hidden,
                num_layers=int(train_args.get("projector_layers", 2)),
                dropout=float(train_args.get("dropout", 0.1)),
            ),
            pdf_figure=ProjectorConfig(
                input_dim=in_dim("pdf_figure_projector"),
                hidden_dim=int(train_args.get("projector_hidden_dim", 1024)),
                output_dim=hidden,
                num_layers=int(train_args.get("projector_layers", 2)),
                dropout=float(train_args.get("dropout", 0.1)),
            ),
            audio=ProjectorConfig(
                input_dim=in_dim("audio_projector"),
                hidden_dim=int(train_args.get("projector_hidden_dim", 1024)),
                output_dim=hidden,
                num_layers=int(train_args.get("projector_layers", 2)),
                dropout=float(train_args.get("dropout", 0.1)),
            ),
            dropout=float(train_args.get("dropout", 0.1)),
        )

        model = AgricultureMultiInputModel.from_pretrained_llm(model_name, cfg)
        model.load_state_dict(state, strict=False)
        model.to(self.device).eval()

        self._ensure_vision_encoder(train_args.get("pdf_feature_model_name", "openai/clip-vit-base-patch32"))
        return model

    def _ensure_vision_encoder(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        if self.vision_model is not None and self.vision_processor is not None:
            return
        self.vision_processor = AutoProcessor.from_pretrained(model_name)
        self.vision_model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def _encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("No images to encode.")
        self._ensure_vision_encoder()
        assert self.vision_processor is not None and self.vision_model is not None

        inputs = self.vision_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if hasattr(self.vision_model, "get_image_features"):
                features = self.vision_model.get_image_features(**inputs)
            else:
                outputs = self.vision_model(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state.mean(dim=1)

        return features

    def _extract_pdf_figures(self, pdf_bytes: bytes, max_figures: int = 2) -> list[Image.Image]:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images: list[Image.Image] = []
        try:
            for page_idx in range(len(document)):
                page = document.load_page(page_idx)
                for image_info in page.get_images(full=True):
                    xref = image_info[0]
                    base_image = document.extract_image(xref)
                    image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                    if image.width < 64 or image.height < 64:
                        continue
                    images.append(image)
                    if len(images) >= max_figures:
                        return images
        finally:
            document.close()
        return images

    def _transcribe_audio_with_groq(self, audio_bytes: bytes, filename: str) -> str:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")

        headers = {"Authorization": f"Bearer {api_key}"}
        data = {
            "model": os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
            "response_format": "json",
            "temperature": "0",
        }
        files = {
            "file": (filename, io.BytesIO(audio_bytes), "application/octet-stream"),
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers=headers,
            data=data,
            files=files,
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("text", "")).strip()

    def _build_text_prompt(self, message: str, history: list[dict[str, str]] | None) -> str:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": "You are an agricultural assistant. Give practical and concise responses.",
            }
        ]
        if history:
            messages.extend(history[-8:])
        messages.append({"role": "user", "content": message})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate_text_only(self, prompt: str) -> str:
        assert self.text_model is not None
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.text_model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0][encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_multimodal(
        self,
        prompt: str,
        crop_image_features: torch.Tensor | None,
        pdf_figure_features: torch.Tensor | None,
    ) -> str:
        assert self.multimodal_model is not None
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        built = self.multimodal_model.build_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            crop_image_features=crop_image_features,
            pdf_figure_features=pdf_figure_features,
            audio_features=None,
        )

        with torch.inference_mode():
            out = self.multimodal_model.llm.generate(
                inputs_embeds=built.inputs_embeds,
                attention_mask=built.attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def chat(
        self,
        message: str,
        history: list[dict[str, str]] | None,
        image_bytes: bytes | None,
        pdf_bytes: bytes | None,
        audio_bytes: bytes | None,
        audio_filename: str,
    ) -> ChatResult:
        transcript = None
        used_modalities: list[str] = []

        if audio_bytes is not None:
            transcript = self._transcribe_audio_with_groq(audio_bytes, audio_filename)
            if transcript:
                message = f"{message}\n\nAudio transcript:\n{transcript}"
                used_modalities.append("audio_text")

        prompt = self._build_text_prompt(message, history)

        crop_image_features = None
        if image_bytes is not None and self.multimodal_model is not None:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            crop_image_features = self._encode_images([image]).unsqueeze(0)
            used_modalities.append("image")

        pdf_figure_features = None
        if pdf_bytes is not None and self.multimodal_model is not None:
            figures = self._extract_pdf_figures(pdf_bytes)
            if figures:
                pdf_figure_features = self._encode_images(figures).unsqueeze(0)
                used_modalities.append("pdf")

        if self.multimodal_model is not None and (crop_image_features is not None or pdf_figure_features is not None):
            answer = self._generate_multimodal(
                prompt=prompt,
                crop_image_features=crop_image_features,
                pdf_figure_features=pdf_figure_features,
            )
        else:
            answer = self._generate_text_only(prompt)

        return ChatResult(answer=answer, transcript=transcript, used_modalities=used_modalities)
