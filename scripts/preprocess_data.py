#!/usr/bin/env python3
"""
Script to run data preprocessing pipeline.

Usage:
    python scripts/preprocess_data.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.load_data import read_tagged_train
from src.data.preprocess import (
    create_train_val_split, 
    prepare_dataset_for_training,
    validate_preprocessed_data,
    create_sample_preprocessed_data
)

def main():
    print("=== eBay NER Data Preprocessing ===")
    
    # Load raw data
    print("Loading raw data...")
    tagged_df = read_tagged_train("data/Tagged_Titles_Train.tsv.gz")
    print(f"Loaded {len(tagged_df)} token records from {tagged_df['Record Number'].nunique()} titles")
    
    # Create train/validation split
    print("\nCreating train/validation split...")
    train_df, val_df = create_train_val_split(tagged_df, val_ratio=0.1, seed=42)
    
    # Prepare datasets for training
    print("\nPreparing datasets for training...")
    train_dataset, val_dataset, label_mappings = prepare_dataset_for_training(
        train_df, 
        val_df,
        config_path="configs/transformer.yml",
        labels_path="configs/labels.txt",
        output_dir="data/processed"
    )
    
    # Validate preprocessed data
    print("\nValidating preprocessed data...")
    validation_results = validate_preprocessed_data(
        train_dataset, 
        val_dataset, 
        label_mappings['label2id']
    )
    
    print("\n=== Validation Results ===")
    for key, value in validation_results.items():
        if key != 'issues':
            print(f"{key}: {value}")
    
    if validation_results['issues']:
        print("\nIssues found:")
        for issue in validation_results['issues']:
            print(f"  - {issue}")
    else:
        print("\n✓ No validation issues found")
    
    # Show sample data
    print("\n=== Sample Preprocessed Data ===")
    create_sample_preprocessed_data(n_samples=3)
    
    print("\n=== Preprocessing Complete ===")
    print("Processed data saved to data/processed/")
    print("Ready for model training!")

if __name__ == "__main__":
    main()
