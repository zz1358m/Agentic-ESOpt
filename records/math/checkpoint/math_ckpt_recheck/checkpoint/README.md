---
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3.5-4B
tags:
  - qwen3-next
  - evolution-strategies
  - math
---

# Qwen3.5-4B Math ES gen025

Hugging Face checkpoint reconstructed from the Dynamic-Agent Math ES run.

- Base runtime checkpoint: `Qwen3.5-4B-text`
- ES scope: full model, 330 tensors / 4,205,751,296 parameters
- Applied updates: 25
- Included generations: 0 through 24
- Update rule: stored ES seeds and z-score-normalized weights, alpha 0.0005
- Weight format: two indexed safetensors shards
- Verified loader: `transformers.AutoModelForCausalLM`

`replay_metadata.json` records the local lineage and `replayed_history.json`
contains the exact 25 update records used to reconstruct this checkpoint.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "zz1358m/qwen35-4b-math-es-gen025"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype="auto")
```
