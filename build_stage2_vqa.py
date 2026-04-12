"""
Build Stage-2 VQA data: image-question-answer JSONL.

This script scans a class-organized image dataset and generates synthetic,
reviewable VQA triplets for multimodal training:
  {"image": "...", "question": "...", "answer": "..."}

Usage:
  python build_stage2_vqa.py \
      --images_dir ./data/plantvillage \
      --output_jsonl ./data/vqa/vqa.jsonl \
      --pairs_per_image 4 \
      --shuffle \
      --max_images 8000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(images_dir: Path) -> List[Path]:
    return [p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def data_dir_hints(cwd: Path, limit: int = 20) -> List[str]:
    data_root = cwd / "data"
    if not data_root.exists() or not data_root.is_dir():
        return []

    dirs = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    return dirs[:limit]


def normalize_label(raw: str) -> str:
    label = raw.strip().replace("___", "|")
    label = label.replace("__", " ").replace("_", " ")
    return " ".join(label.split())


def parse_crop_disease(raw_label: str) -> Tuple[str, str]:
    text = raw_label.strip()
    if "___" in text:
        crop_raw, disease_raw = text.split("___", 1)
        crop = normalize_label(crop_raw).title()
        disease = normalize_label(disease_raw)
        if disease.lower() == "healthy":
            disease = "healthy"
        return crop, disease

    normalized = normalize_label(text)
    if "healthy" in normalized.lower():
        guess_crop = normalized.lower().replace("healthy", "").strip(" -_")
        crop = guess_crop.title() if guess_crop else "Crop"
        return crop, "healthy"

    return "Crop", normalized


def rel_or_abs(path: Path, relative_to: Optional[Path]) -> str:
    if relative_to is None:
        return path.as_posix()

    try:
        return path.resolve().relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_knowledge(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None:
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, str]] = {}
    for disease_name, details in data.items():
        out[disease_name.strip().lower()] = {
            "key_signs": str(details.get("key_signs", "No specific key signs provided.")),
            "management": str(
                details.get(
                    "management",
                    "Use integrated crop management: remove heavily affected tissue, improve airflow, and follow local extension guidance.",
                )
            ),
            "prevention": str(
                details.get(
                    "prevention",
                    "Use resistant varieties, field sanitation, and crop rotation where applicable.",
                )
            ),
        }
    return out


def disease_info(crop: str, disease: str, knowledge: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    if disease.lower() == "healthy":
        return {
            "diagnosis": f"The {crop} plant appears healthy with no clear disease symptoms visible in this image.",
            "severity": "No active disease severity is visible. Continue routine monitoring and preventive care.",
            "management": "No curative treatment is required. Maintain good irrigation, nutrition, and field hygiene.",
            "prevention": "Keep scouting regularly, sanitize tools, and avoid prolonged leaf wetness when possible.",
            "key_signs": "Healthy color and structure are observed without clear lesion or rot patterns.",
        }

    details = knowledge.get(disease.lower(), {})
    key_signs = details.get(
        "key_signs",
        f"Visible signs are consistent with {disease}, but field confirmation is still recommended.",
    )
    management = details.get(
        "management",
        "Apply integrated management: remove severely affected tissue, reduce leaf wetness, improve airflow, and follow local extension recommendations.",
    )
    prevention = details.get(
        "prevention",
        "Use clean planting material, resistant varieties when available, rotation, and preventive monitoring.",
    )

    return {
        "diagnosis": f"The image is consistent with {disease} on {crop}. Confirm with local agronomic inspection if needed.",
        "severity": "Severity appears mild-to-moderate from this view, but precise scoring requires field-level lesion coverage measurement.",
        "management": management,
        "prevention": prevention,
        "key_signs": key_signs,
    }


def generate_pairs(crop: str, disease: str, knowledge: Dict[str, Dict[str, str]]) -> List[Tuple[str, str]]:
    info = disease_info(crop, disease, knowledge)
    return [
        ("What disease is visible in this crop image?", info["diagnosis"]),
        ("How severe does the infection appear?", info["severity"]),
        ("What visual signs support this diagnosis?", info["key_signs"]),
        ("What immediate management steps are recommended?", info["management"]),
        ("How can this be prevented in the next cycle?", info["prevention"]),
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True, help="Root directory with images")
    p.add_argument("--output_jsonl", default="./data/vqa/vqa.jsonl")
    p.add_argument(
        "--relative_to",
        default=".",
        help="Make saved image paths relative to this directory",
    )
    p.add_argument("--class_source", choices=["parent", "stem"], default="parent")
    p.add_argument("--pairs_per_image", type=int, default=4)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument(
        "--knowledge_json",
        default=None,
        help="Optional disease-knowledge JSON with key_signs/management/prevention",
    )
    args = p.parse_args()

    if args.pairs_per_image < 1 or args.pairs_per_image > 5:
        raise ValueError("pairs_per_image must be between 1 and 5")

    images_dir = Path(args.images_dir).expanduser()
    out_path = Path(args.output_jsonl)
    relative_to = Path(args.relative_to) if args.relative_to else None
    knowledge = load_knowledge(Path(args.knowledge_json)) if args.knowledge_json else {}

    if not images_dir.exists():
        cwd = Path.cwd()
        hints = data_dir_hints(cwd)
        hint_text = f"Detected subfolders under ./data: {hints}" if hints else "No ./data subfolders found from current working directory."
        raise FileNotFoundError(
            "images_dir not found: "
            f"{images_dir}\n"
            f"Current working directory: {cwd}\n"
            f"{hint_text}\n"
            "Use --images_dir with the real dataset path (absolute or cwd-relative)."
        )

    images = collect_images(images_dir)
    if not images:
        raise RuntimeError(f"No images found under: {images_dir}")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(images)

    if args.max_images > 0:
        images = images[: args.max_images]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for img in images:
            raw_label = img.parent.name if args.class_source == "parent" else img.stem
            crop, disease = parse_crop_disease(raw_label)

            qa_pairs = generate_pairs(crop, disease, knowledge)
            for question, answer in qa_pairs[: args.pairs_per_image]:
                rec = {
                    "image": rel_or_abs(img, relative_to),
                    "question": question,
                    "answer": answer,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"[done] Wrote {written} VQA samples to: {out_path}")
    print("[note] This dataset is template-generated. Manually review a sample before training.")


if __name__ == "__main__":
    main()
