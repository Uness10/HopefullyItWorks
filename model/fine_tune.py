from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SYSTEM_PROMPT = "You are an expert agriculture assistant. Give practical, concise, field-ready advice."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-file Colab fine-tuning: download agriculture QA datasets, "
            "format instruction data, and train LoRA adapters."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default="outputs/qwen_agri_lora")
    parser.add_argument("--dataset-output-jsonl", default="data/processed/agri_train.jsonl")
    parser.add_argument("--max-samples", type=int, default=8000)
    parser.add_argument("--val-split", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--dataset-name", default="KisanVaani/agriculture-qa-english-only")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answers")
    parser.add_argument("--include-extra-datasets", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dataset-retries", type=int, default=3)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--skip-install", action="store_true")
    return parser.parse_args()


def ensure_dependency(module_name: str, package_spec: str) -> None:
    try:
        importlib.import_module(module_name)
    except ImportError:
        print(f"[deps] Installing {package_spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])


def bootstrap_dependencies(skip_install: bool) -> None:
    if skip_install:
        return

    required = [
        ("datasets", "datasets>=2.18.0"),
        ("transformers", "transformers>=4.40.0"),
        ("peft", "peft>=0.12.0"),
        ("accelerate", "accelerate>=0.33.0"),
    ]
    for module_name, package_spec in required:
        ensure_dependency(module_name, package_spec)


def load_dataset_with_retry(
    load_dataset_fn,
    dataset_name: str,
    split: str,
    token: str | None,
    retries: int,
):
    from datasets import DownloadConfig

    for attempt in range(1, retries + 1):
        try:
            return load_dataset_fn(
                dataset_name,
                split=split,
                token=token,
                download_config=DownloadConfig(max_retries=5),
            )
        except Exception as exc:  # pragma: no cover - network errors are runtime-only.
            if attempt == retries:
                raise
            wait_seconds = 10 * attempt
            print(
                f"[data] Download failed for {dataset_name} (attempt {attempt}/{retries}): {exc}. "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)


def main() -> None:
    args = parse_args()
    bootstrap_dependencies(args.skip_install)

    import torch
    from datasets import concatenate_datasets, load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if args.val_split < 0 or args.val_split >= 1:
        raise ValueError("--val-split must be in [0, 1).")

    set_seed(args.seed)

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if hf_token:
        print("[hf] Using authenticated Hugging Face requests.")
    else:
        print("[hf] No HF token found. Downloads may be slower. Set HF_TOKEN for better reliability.")

    dataset_specs = [
        {
            "name": args.dataset_name,
            "split": args.dataset_split,
            "question_key": args.question_key,
            "answer_key": args.answer_key,
        },
    ]

    if args.include_extra_datasets:
        dataset_specs.extend(
            [
                {
                    "name": "YuvrajSingh9886/Agriculture-Soil-QA-Pairs-Dataset",
                    "split": "train",
                    "question_key": "QUESTION.question",
                    "answer_key": "ANSWER",
                },
                {
                    "name": "YuvrajSingh9886/Agriculture-Irrigation-QA-Pairs-Dataset",
                    "split": "train",
                    "question_key": "QUESTION.question",
                    "answer_key": "ANSWER",
                },
            ]
        )

    normalized_sets = []
    for spec in dataset_specs:
        print(f"[data] Loading {spec['name']} ({spec['split']})")
        ds = load_dataset_with_retry(
            load_dataset,
            dataset_name=spec["name"],
            split=spec["split"],
            token=hf_token,
            retries=args.dataset_retries,
        )

        def _normalize(example: dict) -> dict:
            question = str(example.get(spec["question_key"], "")).strip()
            answer = str(example.get(spec["answer_key"], "")).strip()
            return {"instruction": question, "output": answer}

        ds = ds.map(_normalize, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["instruction"]) > 5 and len(x["output"]) > 2)
        print(f"[data] {spec['name']} usable rows: {len(ds)}")
        normalized_sets.append(ds)

    merged = concatenate_datasets(normalized_sets).shuffle(seed=args.seed)
    if args.max_samples > 0:
        merged = merged.select(range(min(args.max_samples, len(merged))))

    dataset_output_path = Path(args.dataset_output_jsonl)
    dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_json(str(dataset_output_path), lines=True, force_ascii=False)
    print(f"[data] Wrote merged training JSONL: {dataset_output_path}")
    print(f"[data] Total rows: {len(merged)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def _format_record(example: dict) -> dict:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    text_ds = merged.map(_format_record, remove_columns=merged.column_names)

    if args.val_split > 0:
        split_ds = text_ds.train_test_split(test_size=args.val_split, seed=args.seed)
        train_ds = split_ds["train"]
        eval_ds = split_ds["test"]
        print(f"[data] Train rows: {len(train_ds)} | Val rows: {len(eval_ds)}")
    else:
        train_ds = text_ds
        eval_ds = None
        print(f"[data] Train rows: {len(train_ds)} | Val rows: 0 (disabled)")

    def _tokenize(batch: dict) -> dict:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        enc["labels"] = [ids[:] for ids in enc["input_ids"]]
        return enc

    tokenized_train = train_ds.map(_tokenize, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = None
    if eval_ds is not None:
        tokenized_eval = eval_ds.map(_tokenize, batched=True, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        effective_batch = max(1, args.batch_size * args.grad_accum_steps)
        updates_per_epoch = max(1, len(tokenized_train) // effective_batch)
        total_steps = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio) if args.warmup_ratio > 0 else 0

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if tokenized_eval is not None else "no",
        eval_steps=args.save_steps,
        dataloader_pin_memory=torch.cuda.is_available(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_id": args.model_id,
                "output_dir": str(output_dir),
                "dataset_jsonl": str(dataset_output_path),
                "datasets_used": [spec["name"] for spec in dataset_specs],
                "train_rows": len(train_ds),
                "val_rows": 0 if eval_ds is None else len(eval_ds),
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "learning_rate": args.learning_rate,
                "max_seq_length": args.max_seq_length,
            },
            handle,
            indent=2,
        )

    print("[done] LoRA fine-tuning complete.")
    print(f"[done] Adapter + tokenizer saved at: {output_dir}")


if __name__ == "__main__":
    main()