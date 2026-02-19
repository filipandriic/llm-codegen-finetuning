python train_lora.py \
  --dataset /Users/filipandric/codellama_finetune/dataset-ip.jsonl \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir lora_tiny_mps \
  --seq-len 512 \
  --epochs 1 \
  --lr 2e-4 \
  --batch-size 1 \
  --grad-accum 32 \
  --lora-r 8 \
  --lora-alpha 16 \
  --no-packing \
  --grad-checkpointing


python train_lora.py \
  --dataset /Users/filipandric/codellama_finetune/dataset_srb/train.jsonl \
  --base-model deepseek-ai/deepseek-coder-1.3b-instruct \
  --output-dir lora_deepseek13b \
  --seq-len 512 \
  --epochs 1 \
  --lr 2e-4 \
  --batch-size 1 \
  --grad-accum 32 \
  --lora-r 8 \
  --lora-alpha 16 \
  --no-packing \
  --grad-checkpointing
