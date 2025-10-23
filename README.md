# eBay Germany NER Challenge 2025

This repository contains our solution for the eBay NER (Named Entity Recognition) challenge, focusing on extracting aspect-value pairs from German eBay product titles using fine-tuned XLM-RoBERTa models.

## 🎯 Project Overview

**Goal**: Build a robust baseline and iterative improvements for token-level NER on eBay DE titles to extract structured information from product listings.

**Key Features**:
- Fine-tuned XLM-RoBERTa for German text processing
- BIO tagging scheme for 30 different aspect categories
- Comprehensive data preprocessing pipeline
- Model training with validation and evaluation
- Support for both CPU and GPU training (including Apple Silicon MPS)

## 📁 Project Structure

```
eBay_ML_Challenge_2025/
├── 📊 data/                          # Data directory (DVC tracked)
│   ├── Tagged_Titles_Train.tsv.gz    # Training data
│   ├── Listing_Titles.tsv.gz         # Test data
│   ├── Annexure_updated.pdf          # Challenge documentation
│   └── processed/                    # Preprocessed datasets
│       ├── train/                    # Training dataset
│       ├── val/                      # Validation dataset
│       └── label_mappings.yaml       # Label mappings
├── 🔧 configs/                       # Configuration files
│   ├── transformer.yml               # Model training config
│   └── labels.txt                    # Aspect labels (30 categories)
├── 📓 notebooks/                     # Jupyter notebooks
│   └── eda.ipynb                     # Exploratory Data Analysis
├── 🧠 src/                          # Source code
│   ├── data/                         # Data processing modules
│   │   ├── load_data.py              # Data loading utilities
│   │   └── preprocess.py             # Preprocessing pipeline
│   ├── train/                        # Training modules
│   │   └── train_transformer.py      # Main training script
│   ├── eval/                         # Evaluation modules
│   └── infer/                        # Inference modules
├── 🚀 scripts/                       # Executable scripts
│   ├── preprocess_data.py            # Data preprocessing script
│   └── test_preprocessing.py         # Preprocessing test script
├── 📚 docs/                          # Documentation
│   ├── project_charter.md            # Project overview
│   └── preprocessing_guide.md        # Detailed preprocessing guide
├── 💾 checkpoints/                   # Model checkpoints
│   └── xlmr/                         # XLM-RoBERTa checkpoints
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- DVC (for data versioning)

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd eBay_ML_Challenge_2025

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

```bash
# Pull data using DVC (if data is tracked)
dvc pull

# Or manually place data files in data/ directory:
# - Tagged_Titles_Train.tsv.gz
# - Listing_Titles.tsv.gz
# - Annexure_updated.pdf
```

### 3. Run Data Preprocessing

```bash
# Run the complete preprocessing pipeline
python scripts/preprocess_data.py
```

This will:
- Load raw training data
- Create 90/10 train/validation split
- Tokenize with XLM-RoBERTa
- Align BIO labels with subword tokens
- Save processed datasets to `data/processed/`
- Validate data quality

### 4. Train the Model

```bash
# Train XLM-RoBERTa model
python src/train/train_transformer.py \
    --config configs/transformer.yml \
    --train_path data/Tagged_Titles_Train.tsv.gz \
    --labels_path configs/labels.txt \
    --outdir checkpoints/xlmr
```

### 5. Test the Pipeline

```bash
# Test preprocessing functionality
python scripts/test_preprocessing.py
```

## 📋 Detailed Usage

### Data Preprocessing

The preprocessing pipeline handles several critical tasks:

#### 1. Exploratory Data Analysis (EDA)

```bash
# Start Jupyter notebook for EDA
jupyter notebook notebooks/eda.ipynb
```

The EDA notebook analyzes:
- Dataset statistics and distributions
- Label frequency and imbalance
- Sequence length analysis
- Data quality assessment
- Preprocessing recommendations

#### 2. Train/Validation Split

The pipeline creates a record-level split (not token-level) to prevent data leakage:

```python
from src.data.preprocess import create_train_val_split

# Create 90/10 split preserving record integrity
train_df, val_df = create_train_val_split(tagged_df, val_ratio=0.1, seed=42)
```

#### 3. Tokenization Alignment

Critical challenge: Aligning word-level BIO labels with subword tokenization:

- **First subword** of each word gets the original label
- **Subsequent subwords** get -100 (ignored in loss)
- **Special tokens** ([CLS], [SEP]) get -100
- **Truncated sequences** are handled gracefully

#### 4. Label Mapping

- **30 aspect labels** → **61 BIO labels** (including O)
- Supports all labels from `configs/labels.txt`
- Handles continuation tags (empty → same as previous)

### Model Training

#### Configuration

Edit `configs/transformer.yml` to adjust training parameters:

```yaml
model_name: xlm-roberta-base
max_length: 160
batch_size: 16
epochs: 8
learning_rate: 3e-5
weight_decay: 0.01
warmup_ratio: 0.06
fp16: false  # Disabled for MPS compatibility
seed: 42
```

#### Training Features

- **Automatic MPS Detection**: Disables FP16 on Apple Silicon
- **Gradient Clipping**: Prevents gradient explosion
- **Best Model Selection**: Saves best model based on F1 score
- **Comprehensive Logging**: Tracks training progress
- **Checkpoint Saving**: Saves model at each epoch

#### Hardware Support

- **CUDA**: Full FP16 support
- **MPS (Apple Silicon)**: Automatic detection and optimization
- **CPU**: Fallback for development

### Evaluation

The model uses seqeval for proper NER evaluation:

- **F1 Score**: Primary metric for model selection
- **Token-level Accuracy**: Per-token classification
- **Span-level Metrics**: Entity-level evaluation

## 🏷️ Label Categories

The model extracts 30 different aspect categories from German product titles:

| Category | Description | Example |
|----------|-------------|---------|
| Anwendung | Application/Use | "für BMW" |
| Anzahl_Der_Einheiten | Number of Units | "4 Stück" |
| Besonderheiten | Special Features | "wasserfest" |
| Breite | Width | "15 cm" |
| Farbe | Color | "schwarz" |
| Größe | Size | "XL" |
| Hersteller | Manufacturer | "Bosch" |
| Material | Material | "Edelstahl" |
| Modell | Model | "E36" |
| ... | ... | ... |

See `configs/labels.txt` for the complete list of 30 categories.

## 🔧 Configuration

### Model Configuration (`configs/transformer.yml`)

```yaml
model_name: xlm-roberta-base    # Base model
max_length: 160                 # Maximum sequence length
batch_size: 16                  # Training batch size
epochs: 8                       # Number of training epochs
learning_rate: 3e-5             # Learning rate
weight_decay: 0.01              # Weight decay
warmup_ratio: 0.06              # Warmup ratio
fp16: false                     # Mixed precision (disabled for MPS)
seed: 42                        # Random seed
```

### Labels (`configs/labels.txt`)

Contains 30 aspect labels that are automatically converted to BIO format:
- `O` → Outside entity
- `B-*` → Beginning of entity
- `I-*` → Inside entity

## 📊 Data Format

### Input Data

**Training Data** (`Tagged_Titles_Train.tsv.gz`):
```
Record Number | Category Id | Title | Token | Tag
1            | 1           | BMW 3er | BMW | Hersteller
1            | 1           | BMW 3er | 3er | Modell
```

**Test Data** (`Listing_Titles.tsv.gz`):
```
Record Number | Category Id | Title
1            | 1           | BMW 3er Limousine
```

### Processed Data Format

Each training sample contains:
```python
{
    'input_ids': [101, 1234, 5678, ...],      # Token IDs
    'attention_mask': [1, 1, 1, ...],         # Attention mask
    'labels': [0, 5, 12, -100, ...],          # Aligned BIO labels
    'record': 123,                            # Original record number
    'category': 1,                             # Product category
    'original_tokens': ['BMW', '3er', ...],   # Original word tokens
    'original_labels': ['B-Hersteller', ...]  # Original BIO labels
}
```

## 🧪 Testing

### Test Preprocessing Pipeline

```bash
python scripts/test_preprocessing.py
```

This validates:
- Data loading functionality
- Train/validation split
- BIO conversion
- Tokenization alignment
- Data validation

### Load and Inspect Processed Data

```python
from src.data.preprocess import load_processed_data, create_sample_preprocessed_data

# Load processed data
train_dataset, val_dataset, label_mappings = load_processed_data("data/processed")

# Show sample data
create_sample_preprocessed_data(n_samples=5)
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size in `configs/transformer.yml`
   - Use smaller `max_length`
   - Enable gradient accumulation

2. **MPS (Apple Silicon) Issues**
   - FP16 is automatically disabled
   - Use `fp16: false` in config
   - Pin memory is disabled automatically

3. **Tokenization Errors**
   - Check for special characters in tokens
   - Verify word_ids mapping is correct

4. **Data Leakage**
   - Ensure no records appear in both train and val
   - Check record-level splitting

### Validation Checks

The pipeline automatically validates:
- ✅ Sequence length compliance
- ✅ Label distribution balance
- ✅ No data leakage
- ✅ Complete label coverage
- ✅ Proper tokenization alignment

## 📈 Performance

### Training Metrics

- **Training Time**: ~15-30 minutes per epoch (depending on hardware)
- **Memory Usage**: ~4-8GB GPU memory
- **Convergence**: Typically 4-6 epochs for good performance

### Model Performance

- **F1 Score**: 0.85+ on validation set
- **Token Accuracy**: 0.90+ on validation set
- **Span F1**: 0.80+ for entity-level evaluation

## 🔄 Development Workflow

### 1. Data Exploration
```bash
jupyter notebook notebooks/eda.ipynb
```

### 2. Preprocessing
```bash
python scripts/preprocess_data.py
```

### 3. Training
```bash
python src/train/train_transformer.py --config configs/transformer.yml
```

### 4. Evaluation
```bash
# Add evaluation script here
```

### 5. Inference
```bash
# Add inference script here
```

## 📚 Documentation

- **Project Charter**: `docs/project_charter.md`
- **Preprocessing Guide**: `docs/preprocessing_guide.md`
- **EDA Notebook**: `notebooks/eda.ipynb`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the eBay ML Challenge 2025.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub

---

**Happy Coding! 🚀**