This repository contains our code and assets for the eBay NER tagging challenge.
This project contains 
- data/ (datasets, tracked by DVC)
-notebooks (EDA, experiments)
-src/ (source code)
-docs/ (design notes, project charter)
-outputs/ (models, predictions,tracked by DVC)

# eBay Germany NER (Aspect Extraction)

This project fine-tunes **XLM-Roberta** to extract aspect–value pairs
from noisy eBay.de product titles.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
