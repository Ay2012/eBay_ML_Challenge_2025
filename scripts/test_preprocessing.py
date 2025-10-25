#!/usr/bin/env python3
"""
Test script to verify preprocessing pipeline works correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules directly
from src.data.load_data import read_tagged_train, to_bio_sequences, load_label_list, build_bio_maps
from src.data.preprocess import (
    create_train_val_split, 
    tokenize_and_align_labels,
    validate_preprocessed_data
)
from transformers import AutoTokenizer
import yaml

def test_basic_functionality():
    """Test basic preprocessing functionality."""
    print("=== Testing Basic Functionality ===")
    
    # Test data loading
    print("1. Testing data loading...")
    tagged_df = read_tagged_train("data/Tagged_Titles_Train.tsv.gz")
    print(f"   ✓ Loaded {len(tagged_df)} records")
    
    # Test train/val split
    print("2. Testing train/val split...")
    train_df, val_df = create_train_val_split(tagged_df, val_ratio=0.1)
    print(f"   ✓ Train: {len(train_df)} records")
    print(f"   ✓ Validation: {len(val_df)} records")
    
    # Test BIO conversion
    print("3. Testing BIO conversion...")
    train_sequences = to_bio_sequences(train_df)
    print(f"   ✓ Converted {len(train_sequences)} sequences to BIO format")
    
    # Test tokenization
    print("4. Testing tokenization...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    labels = load_label_list("configs/labels.txt")
    _, _, label2id = build_bio_maps(labels)
    
    # Test with first few sequences
    test_sequences = train_sequences[:5]
    tokenized = tokenize_and_align_labels(
        test_sequences, tokenizer, max_length=160, label2id=label2id
    )
    print(f"   ✓ Tokenized {len(tokenized)} sequences")
    
    # Test validation
    print("5. Testing validation...")
    from datasets import Dataset
    train_dataset = Dataset.from_list(tokenized)
    val_dataset = Dataset.from_list(tokenized)  # Using same data for test
    
    validation_results = validate_preprocessed_data(
        train_dataset, val_dataset, label2id
    )
    print(f"   ✓ Validation completed")
    print(f"   ✓ Train samples: {validation_results['train_samples']}")
    print(f"   ✓ Max length: {validation_results['train_max_length']}")
    
    print("\n=== All Tests Passed! ===")
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
        print("\n✓ Preprocessing pipeline is working correctly!")
    except Exception as e:
        print(f"\n✗ Error in preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
