#!/usr/bin/env python3
"""
train_lora.py — LoRA fine-tuning for CodeLlama
Confirmed working with:
  transformers==4.37.2
  trl==0.7.10
  peft==0.8.2
  datasets==2.16.1
  accelerate==0.27.2
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-model", default="codellama/CodeLlama-7b-Instruct-hf")
    ap.add_argument("--output-dir", default="lora_out")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--no-packing", action="store_true")
    ap.add_argument("--grad-checkpointing", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print("[INFO] Loading dataset…")
    ds = load_dataset("json", data_files=args.dataset)["train"]

    print("[INFO] Loading tokenizer + base model…")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    dtype = torch.float32 if torch.backends.mps.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    model.config.use_cache = False

    print("[INFO] Preparing LoRA config…")
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "v_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.log_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=args.grad_checkpointing,
        save_strategy="epoch",
        report_to=[],
    )

    print("[INFO] Starting training…")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        peft_config=peft_cfg,
        max_seq_length=args.seq_len,   # ← fixed
        args=train_args,
        dataset_text_field="text",
        packing=not args.no_packing,
    )

    trainer.train()

    print("[INFO] Saving LoRA adapter to:", args.output_dir)
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("[DONE] Training complete.")

if __name__ == "__main__":
    main()
