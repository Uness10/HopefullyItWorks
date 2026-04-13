"""
Build Stage-1 projector data: image-caption JSONL.

This script scans an image directory and creates records:
  {"image": "path/to/file.jpg", "caption": "..."}

It is designed for class-organized datasets such as PlantVillage where
folder names look like:
  Tomato___Early_blight
  Potato___healthy

Usage:
  python build_stage1_captions.py \
      --images_dir ./data/plantvillage \
      --output_jsonl ./data/captions/captions.jsonl \
      --shuffle \
      --max_images 20000
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


def caption_from_label(crop: str, disease: str) -> str:
    if disease.lower() == "healthy":
        return (
            f"{crop} leaf appears healthy with no clear disease lesions, "
            "chlorosis, or necrotic spotting visible in this image."
        )

    return (
        f"{crop} leaf showing visual symptoms consistent with {disease}. "
        "Visible signs include localized discoloration, lesion patterns, and "
        "affected tissue regions that should be confirmed by field inspection."
    )


def rel_or_abs(path: Path, relative_to: Optional[Path]) -> str:
    if relative_to is None:
        return path.as_posix()

    try:
        return path.resolve().relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_label_map(label_map_json: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if label_map_json is None:
        return {}

    data = json.loads(label_map_json.read_text(encoding="utf-8"))
    mapped: Dict[str, Dict[str, str]] = {}
    for k, v in data.items():
        key = k.strip().lower()
        if isinstance(v, str):
            mapped[key] = {"crop": "Crop", "disease": v}
        else:
            mapped[key] = {
                "crop": str(v.get("crop", "Crop")),
                "disease": str(v.get("disease", "unknown issue")),
            }
    return mapped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True, help="Root directory with images")
    p.add_argument(
        "--output_jsonl",
        default="./data/captions/captions.jsonl",
        help="Where captions.jsonl will be written",
    )
    p.add_argument(
        "--relative_to",
        default=".",
        help="Make saved image paths relative to this directory",
    )
    p.add_argument(
        "--class_source",
        choices=["parent", "stem"],
        default="parent",
        help="How to infer class label: parent folder name or filename stem",
    )
    p.add_argument("--shuffle", action="store_true", help="Shuffle image order")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="If >0, truncate after this many images",
    )
    p.add_argument(
        "--label_map_json",
        default=None,
        help="Optional JSON map for overriding class -> crop/disease",
    )
    args = p.parse_args()

    images_dir = Path(args.images_dir).expanduser()
    out_path = Path(args.output_jsonl)
    relative_to = Path(args.relative_to) if args.relative_to else None
    label_map = load_label_map(Path(args.label_map_json)) if args.label_map_json else {}

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
            key = raw_label.strip().lower()

            if key in label_map:
                crop = label_map[key]["crop"]
                disease = label_map[key]["disease"]
            else:
                crop, disease = parse_crop_disease(raw_label)

            record = {
                "image": rel_or_abs(img, relative_to),
                "caption": caption_from_label(crop, disease),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[done] Wrote {written} caption samples to: {out_path}")


if __name__ == "__main__":
    main()
