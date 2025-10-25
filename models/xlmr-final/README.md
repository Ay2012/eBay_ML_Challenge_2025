# Exported Model

This directory contains the exported model files for inference.

## Files included:
- config.json
- model.safetensors
- tokenizer.json
- tokenizer_config.json
- special_tokens_map.json
- training_args.bin

## Usage:
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("models/xlmr-final")
model = AutoModelForTokenClassification.from_pretrained("models/xlmr-final")
```

## Model Info:
- Model type: XLM-RoBERTa for Token Classification
- Task: Named Entity Recognition (NER)
- Exported from: checkpoints/xlmr
