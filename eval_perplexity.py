#!/usr/bin/env python3
"""
Compute average loss / perplexity on the *assistant portion only* of [INST] samples.

Each JSON line must have {"text": "<s>[INST] ... [/INST]\\n<assistant>...</s>"}.

Usage:
python eval_perplexity.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --data /Users/filipandric/codellama_finetune/dataset/val.jsonl \
  --seq-len 2048

python eval_perplexity.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data /Users/filipandric/codellama_finetune/dataset/val.jsonl \
  --seq-len 2048
"""
import argparse, json, math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

END_INST = '[/INST]\n'
END_SEQ  = '</s>'

def load_jsonl(path):
    for ln in Path(path).read_text(encoding='utf-8').splitlines():
        ln = ln.strip()
        if ln:
            yield json.loads(ln)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--seq-len', type=int, default=2048)
    ap.add_argument('--max-samples', type=int, default=100000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map='auto' if torch.cuda.is_available() else None
    )
    model.eval()

    total_loss, total_tgt_tokens = 0.0, 0
    with torch.no_grad():
        for i, ex in enumerate(load_jsonl(args.data)):
            if i >= args.max_samples: break
            text = ex.get('text', '')
            if not text: continue

            # Split prompt vs target by the end-of-instruction marker
            pos = text.find(END_INST)
            if pos == -1:  # skip malformed
                continue
            prompt = text[:pos + len(END_INST)]
            target = text[pos + len(END_INST):]

            # Optional: strip closing </s> if present in target
            if target.endswith(END_SEQ):
                target = target[:-len(END_SEQ)]

            # Tokenize full and prompt to align labels
            full_ids = tok(prompt + target, return_tensors='pt', truncation=True, max_length=args.seq_len)
            prompt_ids = tok(prompt, return_tensors='pt', truncation=True, max_length=args.seq_len)

            input_ids = full_ids['input_ids'][0]
            attn_mask = full_ids['attention_mask'][0]
            plen = prompt_ids['input_ids'].shape[1]
            seqlen = input_ids.shape[0]
            if plen >= seqlen:
                continue  # target truncated away

            labels = input_ids.clone()
            labels[:plen] = -100  # mask prompt tokens; only score target

            batch = {'input_ids': input_ids.unsqueeze(0),
                     'attention_mask': attn_mask.unsqueeze(0),
                     'labels': labels.unsqueeze(0)}
            for k in batch:
                batch[k] = batch[k].to(model.device)

            out = model(**batch)
            # average loss over *unmasked* tokens; HF returns mean over all non -100
            loss = out.loss.item()
            # estimate unmasked tokens:
            tgt_tokens = int((labels != -100).sum().item())
            total_loss += loss * tgt_tokens
            total_tgt_tokens += tgt_tokens

    if total_tgt_tokens == 0:
        print('[RESULT] No target tokens evaluated.')
        return

    avg_loss = total_loss / total_tgt_tokens
    ppl = math.exp(avg_loss)
    print(f"[RESULT] tgt_tokens={total_tgt_tokens}  avg_loss={avg_loss:.6f}  perplexity={ppl:.3f}")

if __name__ == '__main__':
    main()
