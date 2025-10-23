#!/usr/bin/env python3
"""
Export a clean model for sharing - removes training artifacts and keeps only inference files.
"""

import argparse
import shutil
import os
from pathlib import Path

def export_clean_model(checkpoint_dir: str, output_dir: str):
    """Export only the files needed for model inference."""
    
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Files needed for inference
    inference_files = [
        "config.json",
        "model.safetensors", 
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "training_args.bin"  # Optional but useful for reproducibility
    ]
    
    print(f"Exporting clean model from {checkpoint_dir} to {output_dir}")
    
    copied_files = []
    for file_name in inference_files:
        src_file = checkpoint_path / file_name
        if src_file.exists():
            dst_file = output_path / file_name
            shutil.copy2(src_file, dst_file)
            copied_files.append(file_name)
            print(f"✓ Copied {file_name}")
        else:
            print(f"⚠️  {file_name} not found in checkpoint")
    
    # Create a README for the exported model
    readme_content = f"""# Exported Model

This directory contains the exported model files for inference.

## Files included:
{chr(10).join(f"- {f}" for f in copied_files)}

## Usage:
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
model = AutoModelForTokenClassification.from_pretrained("{output_dir}")
```

## Model Info:
- Model type: XLM-RoBERTa for Token Classification
- Task: Named Entity Recognition (NER)
- Exported from: {checkpoint_dir}
"""
    
    with open(output_path / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"\n✅ Clean model exported to: {output_dir}")
    print(f"📁 Files copied: {len(copied_files)}")
    print(f"📄 README.md created with usage instructions")

def main():
    parser = argparse.ArgumentParser(description="Export clean model for sharing")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                       help="Path to checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Path to output directory for clean model")
    
    args = parser.parse_args()
    
    try:
        export_clean_model(args.checkpoint_dir, args.output_dir)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
