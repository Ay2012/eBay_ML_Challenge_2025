# Data Preprocessing Guide

This guide explains the data preprocessing pipeline for the eBay NER challenge.

## Overview

The preprocessing pipeline handles:
1. **Exploratory Data Analysis (EDA)** - Understanding the dataset
2. **Train/Validation Split** - Random split preserving record integrity
3. **Tokenization Alignment** - Aligning BIO labels with XLM-RoBERTa subwords
4. **Dataset Preparation** - Creating HuggingFace datasets for training
5. **Data Validation** - Quality checks and validation

## Files Structure

```
├── notebooks/
│   └── eda.ipynb                    # Comprehensive EDA notebook
├── src/data/
│   ├── load_data.py                 # Data loading utilities
│   └── preprocess.py                # Preprocessing functions
├── scripts/
│   ├── preprocess_data.py          # Main preprocessing script
│   └── test_preprocessing.py       # Test script
└── data/processed/                  # Output directory
    ├── train/                       # Training dataset
    ├── val/                        # Validation dataset
    └── label_mappings.yaml        # Label mappings
```

## Quick Start

### 1. Run EDA (Optional but Recommended)

```bash
# Start Jupyter notebook
jupyter notebook notebooks/eda.ipynb
```

This will analyze:
- Dataset statistics and distributions
- Label frequency and imbalance
- Sequence length analysis
- Data quality assessment
- Preprocessing recommendations

### 2. Run Preprocessing Pipeline

```bash
# Run the complete preprocessing pipeline
python scripts/preprocess_data.py
```

This will:
- Load raw training data
- Create 90/10 train/validation split
- Tokenize with XLM-RoBERTa
- Align BIO labels with subword tokens
- Save processed datasets
- Validate data quality

### 3. Test Preprocessing

```bash
# Test the preprocessing pipeline
python scripts/test_preprocessing.py
```

## Key Features

### Tokenization Alignment

The pipeline handles the critical challenge of aligning word-level BIO labels with subword tokenization:

- **First subword** of each word gets the original label
- **Subsequent subwords** get -100 (ignored in loss)
- **Special tokens** ([CLS], [SEP]) get -100
- **Truncated sequences** are handled gracefully

### Label Mapping

- **30 aspect labels** → **61 BIO labels** (including O)
- Supports all labels from `configs/labels.txt`
- Handles continuation tags (empty → same as previous)

### Data Validation

The pipeline includes comprehensive validation:
- Sequence length analysis
- Label distribution checks
- Data leakage detection
- Missing label identification

## Configuration

### Model Configuration (`configs/transformer.yml`)

```yaml
model_name: xlm-roberta-base
max_length: 160
batch_size: 16
epochs: 8
learning_rate: 3e-5
```

### Labels (`configs/labels.txt`)

Contains 30 aspect labels:
- Anwendung, Anzahl_Der_Einheiten, Besonderheiten, etc.
- Automatically converted to BIO format (B-*, I-*, O)

## Usage Examples

### Load Processed Data

```python
from src.data.preprocess import load_processed_data

# Load preprocessed datasets
train_dataset, val_dataset, label_mappings = load_processed_data("data/processed")

print(f"Train samples: {len(train_dataset)}")
print(f"Labels: {label_mappings['num_labels']}")
```

### Custom Preprocessing

```python
from src.data.preprocess import (
    create_train_val_split, 
    tokenize_and_align_labels,
    prepare_dataset_for_training
)

# Load raw data
tagged_df = read_tagged_train("data/Tagged_Titles_Train.tsv.gz")

# Custom split
train_df, val_df = create_train_val_split(tagged_df, val_ratio=0.15, seed=123)

# Prepare datasets
train_dataset, val_dataset, mappings = prepare_dataset_for_training(
    train_df, val_df, output_dir="data/custom_processed"
)
```

### Validate Data

```python
from src.data.preprocess import validate_preprocessed_data

# Validate preprocessed data
results = validate_preprocessed_data(train_dataset, val_dataset, label2id)

print(f"Max length: {results['train_max_length']}")
print(f"Issues: {results['issues']}")
```

## Output Format

### Processed Dataset Structure

Each sample contains:
```python
{
    'input_ids': [101, 1234, 5678, ...],      # Token IDs
    'attention_mask': [1, 1, 1, ...],         # Attention mask
    'labels': [0, 5, 12, -100, ...],          # Aligned BIO labels
    'record': 123,                            # Original record number
    'category': 1,                             # Product category
    'original_tokens': ['MINI', '1.6', ...],  # Original word tokens
    'original_labels': ['B-Hersteller', ...]  # Original BIO labels
}
```

### Label Mappings

```python
{
    'id2label': {0: 'O', 1: 'B-Anwendung', 2: 'I-Anwendung', ...},
    'label2id': {'O': 0, 'B-Anwendung': 1, 'I-Anwendung': 2, ...},
    'bio_labels': ['O', 'B-Anwendung', 'I-Anwendung', ...],
    'num_labels': 61
}
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or use smaller max_length
2. **Tokenization Errors**: Check for special characters in tokens
3. **Label Alignment**: Verify word_ids mapping is correct
4. **Data Leakage**: Ensure no records appear in both train and val

### Validation Checks

The pipeline automatically checks for:
- ✅ Sequence length compliance
- ✅ Label distribution balance
- ✅ No data leakage
- ✅ Complete label coverage
- ✅ Proper tokenization alignment

## Next Steps

After preprocessing:
1. **Train Model**: Use processed data for transformer training
2. **Evaluate**: Test on validation set
3. **Inference**: Apply to test data for predictions

The preprocessed data is ready for training with HuggingFace Transformers!
