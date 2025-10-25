"""
Data preprocessing module for eBay NER task.

This module handles:
- Train/validation splitting
- Tokenization alignment with XLM-RoBERTa
- Dataset preparation for training
- Data validation and quality checks
"""

import os
import random
from typing import List, Dict, Tuple, Optional
from collections import Counter
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
import torch
from tqdm import tqdm
import yaml

from .load_data import read_tagged_train, to_bio_sequences, load_label_list, build_bio_maps


def create_train_val_split(
    tagged_df: pd.DataFrame, 
    val_ratio: float = 0.1, 
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation split preserving record-level integrity.
    
    Args:
        tagged_df: DataFrame with tagged training data
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Get unique record numbers
    record_numbers = tagged_df['Record Number'].unique()
    n_records = len(record_numbers)
    n_val = int(n_records * val_ratio)
    
    # Randomly sample validation records
    val_records = set(random.sample(list(record_numbers), n_val))
    train_records = set(record_numbers) - val_records
    
    # Split the dataframe
    train_df = tagged_df[tagged_df['Record Number'].isin(train_records)].copy()
    val_df = tagged_df[tagged_df['Record Number'].isin(val_records)].copy()
    
    print(f"Train records: {len(train_records)} ({len(train_records)/n_records*100:.1f}%)")
    print(f"Validation records: {len(val_records)} ({len(val_records)/n_records*100:.1f}%)")
    
    return train_df, val_df


def tokenize_and_align_labels(
    sequences: List[Dict], 
    tokenizer, 
    max_length: int = 160,
    label2id: Dict[str, int] = None
) -> List[Dict]:
    """
    Tokenize sequences and align BIO labels with subword tokens.
    
    Args:
        sequences: List of sequence dicts with 'tokens' and 'bio_labels'
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        label2id: Label to ID mapping
        
    Returns:
        List of tokenized sequences with aligned labels
    """
    tokenized_sequences = []
    
    for seq in tqdm(sequences, desc="Tokenizing sequences"):
        tokens = seq['tokens']
        bio_labels = seq['bio_labels']
        
        # Tokenize the sequence
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        
        # Get word IDs for alignment
        word_ids = encoding.word_ids()
        
        # Align labels with subword tokens
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                # Special tokens ([CLS], [SEP], etc.) get -100 (ignored in loss)
                aligned_labels.append(-100)
            else:
                # Get the original label for this word
                if word_id < len(bio_labels):
                    label = bio_labels[word_id]
                    if label2id and label in label2id:
                        aligned_labels.append(label2id[label])
                    else:
                        aligned_labels.append(-100)  # Unknown label
                else:
                    aligned_labels.append(-100)  # Out of bounds
        
        tokenized_sequences.append({
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': aligned_labels,
            'record': seq['record'],
            'category': seq['category'],
            'original_tokens': tokens,
            'original_labels': bio_labels
        })
    
    return tokenized_sequences


def prepare_dataset_for_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config_path: str = "configs/transformer.yml",
    labels_path: str = "configs/labels.txt",
    output_dir: str = "data/processed"
) -> Tuple[Dataset, Dataset, Dict]:
    """
    Prepare complete dataset for training with tokenization and alignment.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe  
        config_path: Path to model config
        labels_path: Path to labels file
        output_dir: Directory to save processed data
        
    Returns:
        Tuple of (train_dataset, val_dataset, label_mappings)
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load labels and create mappings
    labels = load_label_list(labels_path)
    bio_labels, id2label, label2id = build_bio_maps(labels)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Convert to BIO sequences
    print("Converting to BIO sequences...")
    train_sequences = to_bio_sequences(train_df)
    val_sequences = to_bio_sequences(val_df)
    
    # Tokenize and align labels
    print("Tokenizing and aligning labels...")
    train_tokenized = tokenize_and_align_labels(
        train_sequences, tokenizer, config['max_length'], label2id
    )
    val_tokenized = tokenize_and_align_labels(
        val_sequences, tokenizer, config['max_length'], label2id
    )
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_tokenized)
    val_dataset = Dataset.from_list(val_tokenized)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed datasets
    train_dataset.save_to_disk(os.path.join(output_dir, "train"))
    val_dataset.save_to_disk(os.path.join(output_dir, "val"))
    
    # Save label mappings
    label_mappings = {
        'id2label': id2label,
        'label2id': label2id,
        'bio_labels': bio_labels,
        'num_labels': len(bio_labels)
    }
    
    with open(os.path.join(output_dir, "label_mappings.yaml"), 'w') as f:
        yaml.dump(label_mappings, f)
    
    print(f"Processed datasets saved to {output_dir}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of labels: {len(bio_labels)}")
    
    return train_dataset, val_dataset, label_mappings


def validate_preprocessed_data(
    train_dataset: Dataset,
    val_dataset: Dataset,
    label2id: Dict[str, int],
    max_length: int = 160
) -> Dict[str, any]:
    """
    Validate preprocessed data for common issues.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        label2id: Label to ID mapping
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'issues': []
    }
    
    # Check sequence lengths
    train_lengths = [len(seq['input_ids']) for seq in train_dataset]
    val_lengths = [len(seq['input_ids']) for seq in val_dataset]
    
    results['train_max_length'] = max(train_lengths)
    results['val_max_length'] = max(val_lengths)
    results['train_avg_length'] = np.mean(train_lengths)
    results['val_avg_length'] = np.mean(val_lengths)
    
    # Check for sequences exceeding max_length
    train_long = sum(1 for l in train_lengths if l >= max_length)
    val_long = sum(1 for l in val_lengths if l >= max_length)
    
    if train_long > 0:
        results['issues'].append(f"Train: {train_long} sequences at max_length")
    if val_long > 0:
        results['issues'].append(f"Validation: {val_long} sequences at max_length")
    
    # Check label distribution
    train_labels = []
    val_labels = []
    
    for seq in train_dataset:
        train_labels.extend([l for l in seq['labels'] if l != -100])
    for seq in val_dataset:
        val_labels.extend([l for l in seq['labels'] if l != -100])
    
    train_label_counts = Counter(train_labels)
    val_label_counts = Counter(val_labels)
    
    results['train_label_dist'] = dict(train_label_counts)
    results['val_label_dist'] = dict(val_label_counts)
    
    # Check for missing labels
    all_label_ids = set(label2id.values())
    train_label_ids = set(train_label_counts.keys())
    val_label_ids = set(val_label_counts.keys())
    
    missing_train = all_label_ids - train_label_ids
    missing_val = all_label_ids - val_label_ids
    
    if missing_train:
        results['issues'].append(f"Missing labels in train: {missing_train}")
    if missing_val:
        results['issues'].append(f"Missing labels in validation: {missing_val}")
    
    # Check for data leakage (same records in train and val)
    train_records = set(seq['record'] for seq in train_dataset)
    val_records = set(seq['record'] for seq in val_dataset)
    overlap = train_records & val_records
    
    if overlap:
        results['issues'].append(f"Data leakage: {len(overlap)} records in both train and val")
    
    return results


def load_processed_data(data_dir: str = "data/processed") -> Tuple[Dataset, Dataset, Dict]:
    """
    Load previously processed datasets.
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        Tuple of (train_dataset, val_dataset, label_mappings)
    """
    train_dataset = Dataset.load_from_disk(os.path.join(data_dir, "train"))
    val_dataset = Dataset.load_from_disk(os.path.join(data_dir, "val"))
    
    with open(os.path.join(data_dir, "label_mappings.yaml"), 'r') as f:
        label_mappings = yaml.safe_load(f)
    
    return train_dataset, val_dataset, label_mappings


def create_sample_preprocessed_data(
    n_samples: int = 5,
    data_dir: str = "data/processed"
) -> None:
    """
    Create sample preprocessed data for inspection.
    
    Args:
        n_samples: Number of samples to show
        data_dir: Directory containing processed data
    """
    train_dataset, val_dataset, label_mappings = load_processed_data(data_dir)
    
    print("=== Sample Preprocessed Data ===")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of labels: {label_mappings['num_labels']}")
    
    print(f"\n=== Sample Training Examples ===")
    for i in range(min(n_samples, len(train_dataset))):
        sample = train_dataset[i]
        print(f"\nExample {i+1}:")
        print(f"  Record: {sample['record']}")
        print(f"  Category: {sample['category']}")
        print(f"  Input length: {len(sample['input_ids'])}")
        print(f"  Original tokens: {sample['original_tokens'][:10]}{'...' if len(sample['original_tokens']) > 10 else ''}")
        print(f"  Original labels: {sample['original_labels'][:10]}{'...' if len(sample['original_labels']) > 10 else ''}")
        
        # Show tokenization alignment
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
        labels = [label_mappings['id2label'].get(l, 'IGNORED') for l in sample['labels']]
        
        print(f"  Tokenized: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        print(f"  Aligned labels: {labels[:15]}{'...' if len(labels) > 15 else ''}")


if __name__ == "__main__":
    # Example usage
    print("Loading and preprocessing data...")
    
    # Load raw data
    tagged_df = read_tagged_train("data/Tagged_Titles_Train.tsv.gz")
    
    # Create train/val split
    train_df, val_df = create_train_val_split(tagged_df)
    
    # Prepare datasets
    train_dataset, val_dataset, label_mappings = prepare_dataset_for_training(
        train_df, val_df
    )
    
    # Validate data
    validation_results = validate_preprocessed_data(
        train_dataset, val_dataset, label_mappings['label2id']
    )
    
    print("\n=== Validation Results ===")
    for key, value in validation_results.items():
        if key != 'issues':
            print(f"{key}: {value}")
    
    if validation_results['issues']:
        print("\nIssues found:")
        for issue in validation_results['issues']:
            print(f"  - {issue}")
    
    # Show sample data
    create_sample_preprocessed_data()
