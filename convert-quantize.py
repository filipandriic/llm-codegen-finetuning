python3 ./convert_hf_to_gguf.py \
  /Users/filipandric/codellama_finetune/tinyllama_finetuned_hf \
  --outfile /Users/filipandric/codellama_finetuned.gguf

  ./bin/llama-quantize \
  /Users/filipandric/codellama_finetuned.gguf \
  /Users/filipandric/codellama_finetuned.Q4_K_M.gguf \
  Q4_K_M


python3 ./convert_hf_to_gguf.py \
  /Users/filipandric/codellama_finetune/deepseek_finetuned \
  --outfile /Users/filipandric/deepseek_finetuned.gguf

./bin/llama-quantize \
  /Users/filipandric/deepseek_finetuned.gguf \
  /Users/filipandric/deepseek_finetuned.Q4_K_M.gguf \
  Q4_K_M