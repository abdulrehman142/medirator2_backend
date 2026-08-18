"""QLoRA fine-tuning skeleton for Medirator hospital-domain adaptation.

This script is intentionally optional. Prompt-only RAG already works via Ollama.
Run QLoRA only if you have a CUDA GPU and install finetune extras.

Example:
  pip install -r finetune/requirements.txt
  python finetune/train_qlora.py --base-model meta-llama/Llama-3.2-3B-Instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Medirator QLoRA trainer (GPU required)")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--train-file", default="finetune/data/train.jsonl")
    parser.add_argument("--output-dir", default="finetune/outputs/medirator-qlora")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    train_path = Path(args.train_file)
    if not train_path.exists():
        raise SystemExit(f"Missing training file: {train_path}")

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
    except ImportError as exc:
        raise SystemExit(
            "Fine-tune dependencies missing. Install with:\n"
            "  pip install -r finetune/requirements.txt\n"
            f"Original error: {exc}"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not available. QLoRA training requires a GPU. "
            "Prompt-only Ollama RAG remains the supported local path."
        )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        ),
    )

    ds = load_dataset("json", data_files={"train": str(train_path)})

    def format_row(row):
        text = (
            f"### Instruction:\n{row['instruction']}\n\n"
            f"### Input:\n{row['input']}\n\n"
            f"### Response:\n{row['output']}"
        )
        tokens = tokenizer(text, truncation=True, max_length=1024)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    train_ds = ds["train"].map(format_row, remove_columns=ds["train"].column_names)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_steps=200,
            report_to=[],
        ),
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
