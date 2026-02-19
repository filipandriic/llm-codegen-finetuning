#!/usr/bin/env python3
"""
merge_lora.py — merge a LoRA adapter into its base model and save a standard HF model.

Usage:
  python merge_lora.py \
    --base deepseek-ai/deepseek-coder-1.3b-instruct \
    --adapter lora_deepseek13b \
    --out deepseek_finetuned_hf \
    --dtype fp32
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base HF model id or path")
    ap.add_argument("--adapter", required=True, help="LoRA adapter folder (output of train_lora.py)")
    ap.add_argument("--out", required=True, help="Output directory for merged HF model")
    ap.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="fp32")
    ap.add_argument("--no-tokenizer", dest="no_tokenizer", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[INFO] Loading base: {args.base}")
    dtype = DTYPE_MAP[args.dtype]
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map=None,   # set to 'auto' if you want GPU offload
    )
    base.config.use_cache = False

    print(f"[INFO] Applying LoRA from: {args.adapter}")
    merged = PeftModel.from_pretrained(base, args.adapter)
    merged = merged.merge_and_unload()  # -> plain HF model

    print(f"[INFO] Saving merged model to: {args.out}")
    merged.save_pretrained(args.out)

    if not args.no_tokenizer:
        print("[INFO] Saving tokenizer…")
        tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
        tok.save_pretrained(args.out)

    print("[DONE] Merge complete.")

if __name__ == "__main__":
    main()
