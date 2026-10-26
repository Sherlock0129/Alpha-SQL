"""Single-GPU LoRA supervised fine-tuning for Qwen2.5-Coder-7B-Instruct.

Input is JSONL. Each line has a ``messages`` array whose last message is the
assistant answer to train on. Earlier messages become the prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALLOWED_ROLES = {"system", "user", "assistant"}


def read_examples(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)

    examples: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc

            messages = item.get("messages") if isinstance(item, dict) else None
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_number}: expected at least two messages")
            if any(
                not isinstance(message, dict)
                or message.get("role") not in ALLOWED_ROLES
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
                for message in messages
            ):
                raise ValueError(f"{path}:{line_number}: invalid role or content")
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"{path}:{line_number}: final message must be assistant")
            if not any(message["role"] == "user" for message in messages[:-1]):
                raise ValueError(f"{path}:{line_number}: prompt needs a user message")

            clean = [
                {"role": message["role"], "content": message["content"]}
                for message in messages
            ]
            examples.append({"prompt": clean[:-1], "completion": [clean[-1]]})

    if not examples:
        raise ValueError(f"{path}: no training examples")
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_examples(args.train_jsonl)
    eval_rows = read_examples(args.eval_jsonl) if args.eval_jsonl else None
    if args.max_length <= 0 or args.lora_r <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("max-length, lora-r and gradient-accumulation-steps must be positive")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; submit this script to a GPU node")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training configuration requires BF16 GPU support")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = SFTConfig(
        output_dir=str(args.output_dir),
        model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        max_length=args.max_length,
        completion_only_loss=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows) if eval_rows else None,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    if eval_rows:
        print(trainer.evaluate())
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
