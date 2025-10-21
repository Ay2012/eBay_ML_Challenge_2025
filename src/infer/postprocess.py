import argparse
import os
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.data.load_data import read_listings, whitespace_tokens, bio_to_spans

def predict_bio(tokens: List[str], tokenizer, model, id2label, max_length: int = 160) -> List[str]:
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits[0]
        pred_ids = logits.argmax(-1).cpu().tolist()
    word_ids = enc.word_ids(0)
    bio = []
    seen = set()
    for i, w in enumerate(word_ids):
        if w is None:
            continue
        if w not in seen:  # first subword only
            bio.append(id2label[int(pred_ids[i])])
            seen.add(w)
    return bio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--listings", type=str, default="data/raw/Listing_Titles.tsv.gz")
    parser.add_argument("--out", type=str, default="data/submissions/quiz_xlmr.tsv")
    parser.add_argument("--start_record", type=int, default=5001)
    parser.add_argument("--end_record", type=int, default=30000)
    parser.add_argument("--max_length", type=int, default=160)
    args = parser.parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    id2label = model.config.id2label

    # Load listings and slice quiz range
    df = read_listings(args.listings)
    mask = (df["Record Number"].astype(int) >= args.start_record) & (df["Record Number"].astype(int) <= args.end_record)
    quiz = df.loc[mask].copy()

    rows = []
    for _, row in quiz.iterrows():
        rec = row["Record Number"]
        cat = row["Category Id"]
        title = row["Title"]
        toks = whitespace_tokens(title)
        bio = predict_bio(toks, tokenizer, model, id2label, args.max_length)
        spans = bio_to_spans(toks, bio)
        for tag, value in spans:
            if tag == "O":
                continue  # you may include O; score unchanged. We skip.
            rows.append([rec, cat, tag, value])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows, columns=["Record Number", "Category Id", "Aspect Name", "Aspect Value"]) \
      .to_csv(args.out, sep="\t", index=False)
    print("Wrote submission TSV:", args.out)

if __name__ == "__main__":
    main()
