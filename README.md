# Domain-Specific Code Generation via Fine-Tuned LLMs

This project implements an end-to-end fine-tuning pipeline for adapting small-scale LLMs 
(TinyLlama 1.3B and DeepSeek 1.1B) to domain-specific code generation tasks.

## Features
- Automated dataset construction from project repositories
- LoRA-based fine-tuning
- Model merging and evaluation (perplexity)
- GGUF conversion and quantization
- llama.cpp inference support

## Models Compared
- TinyLlama 1.3B
- DeepSeek 1.1B

## Pipeline
make dataset → train → merge → eval → convert → quant

