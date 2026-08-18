# Medirator QLoRA Fine-Tuning

Prompt-only RAG (Grok / xAI) is the default runtime path.

This folder supports assignment §2.5:

1. `data/train.jsonl` / `data/eval.jsonl` — instruction pairs built from the synthetic KB (SOAP + domain templates)
2. `evaluate_baseline.py` — checks gold template key compliance (no GPU)
3. `train_qlora.py` — QLoRA trainer (CUDA GPU required)

```bash
cd backend
pip install -r finetune/requirements.txt
python finetune/evaluate_baseline.py
python finetune/train_qlora.py
```

After training, point a local OpenAI-compatible / llama.cpp / Ollama Modelfile at the adapter (or merge weights) and compare SOAP consistency against prompt-only `llama3.2`.
