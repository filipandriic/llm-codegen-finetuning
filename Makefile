SHELL := /bin/bash
.ONESHELL:

include pipeline.conf

.PHONY: all dataset train merge eval convert quant clean help

# run everything; 'clean' now only removes LoRA + the unquantized GGUF
all: dataset train merge eval convert quant clean

help:
	@echo "Targets:"
	@echo "  make dataset   # build train/val from $(ROOTS) -> $(DATASET_DIR)"
	@echo "  make train     # train LoRA: base=$(BASE_MODEL) out=$(LORA_OUT)"
	@echo "  make merge     # merge LoRA -> HF folder in $(MERGED_HF)"
	@echo "  make eval      # evaluate perplexity (assistant-only) -> $(ART_DIR)/eval_ppl.txt"
	@echo "  make convert   # convert HF -> GGUF -> $(GGUF_OUT)"
	@echo "  make quant     # quantize GGUF -> $(QUANT_OUT) with $(QUANT_METHOD)"
	@echo "  make clean     # remove LoRA + intermediate GGUF (keeps HF, datasets, quantized GGUF)"

dataset:
	@echo "[DATASET] building into $(DATASET_DIR)"
	mkdir -p "$(DATASET_DIR)"
	$(MAKE_DATASET) \
	  --roots $(ROOTS) \
	  --out "$(DATASET_DIR)" \
	  --modes $(DATASET_MODES) \
	  --samples-per-project $(SAMPLES_PER_PROJECT) \
	  --max-relevant-files $(MAX_RELEVANT_FILES) \
	  --max-file-bytes $(MAX_FILE_BYTES) \
	  --val-frac $(VAL_FRAC) \
	  --format $(FORMAT)

train:
	@echo "[TRAIN] base=$(BASE_MODEL) out=$(LORA_OUT)"
	test -f "$(TRAIN_DATA)" || { echo "ERROR: dataset not found at $(TRAIN_DATA). Run 'make dataset' first."; exit 1; }
	$(TRAIN) \
	  --dataset "$(TRAIN_DATA)" \
	  --base-model "$(BASE_MODEL)" \
	  --output-dir "$(LORA_OUT)" \
	  --seq-len $(SEQ_LEN) \
	  --epochs $(EPOCHS) \
	  --lr $(LR) \
	  --batch-size $(BATCH_SIZE) \
	  --grad-accum $(GRAD_ACCUM) \
	  --lora-r $(LORA_R) \
	  --lora-alpha $(LORA_ALPHA) \
	  $(NO_PACKING) \
	  $(GRAD_CHECKPOINTING)

merge:
	@echo "[MERGE] base=$(MERGE_BASE) adapter=$(MERGE_ADAPTER) -> $(MERGED_HF)"
	test -d "$(LORA_OUT)" || { echo "ERROR: LoRA folder '$(LORA_OUT)' not found. Run 'make train' first."; exit 1; }
	mkdir -p "$(ART_DIR)"
	$(MERGE) \
	  --base "$(MERGE_BASE)" \
	  --adapter "$(MERGE_ADAPTER)" \
	  --out "$(MERGED_HF)" \
	  --dtype fp32

# -------------------- NEW: evaluation step --------------------
eval:
	@echo "[EVAL] perplexity (assistant-only) on $(VAL_DATA)"
	test -f "$(VAL_DATA)" || { echo "ERROR: $(VAL_DATA) not found. Run 'make dataset' first."; exit 1; }
	test -d "$(MERGED_HF)" || { echo "ERROR: merged HF model '$(MERGED_HF)' not found. Run 'make merge' first."; exit 1; }
	mkdir -p "$(ART_DIR)"
	$(EVAL_PPL) --model "$(MERGED_HF)" --data "$(VAL_DATA)" --seq-len 2048 | tee "$(ART_DIR)/eval_ppl.txt"
	@echo "[EVAL] wrote $(ART_DIR)/eval_ppl.txt"

convert:
	@echo "[CONVERT] $(MERGED_HF) -> $(GGUF_OUT)"
	test -d "$(MERGED_HF)" || { echo "ERROR: merged HF model folder '$(MERGED_HF)' not found. Run 'make merge' first."; exit 1; }
	mkdir -p "$(ART_DIR)"
	python "$(LLAMACPP_DIR)/convert_hf_to_gguf.py" \
	  "$(MERGED_HF)" \
	  --outfile "$(GGUF_OUT)"

quant:
	@echo "[QUANT] $(GGUF_OUT) -> $(QUANT_OUT) [$(QUANT_METHOD)]"
	test -f "$(GGUF_OUT)" || { echo "ERROR: GGUF file '$(GGUF_OUT)' not found. Run 'make convert' first."; exit 1; }
	"$(LLAMACPP_DIR)/build/bin/llama-quantize" \
	  "$(GGUF_OUT)" \
	  "$(QUANT_OUT)" \
	  "$(QUANT_METHOD)"

# keep datasets, merged HF, and final quantized GGUF
clean:
	@echo "[CLEAN] removing intermediate artifacts (keeps HF, datasets, final .gguf)"
	rm -rf "$(LORA_OUT)" "$(GGUF_OUT)" 2>/dev/null || true
