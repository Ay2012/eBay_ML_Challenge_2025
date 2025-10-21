import argparse
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
import pandas as pd
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import f1_score

from src.data.load_data import (
    read_tagged_train,
    to_bio_sequences,
    load_label_list,
    build_bio_maps,
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, encs: List[Dict]):
        self.encs = encs
    def __len__(self):
        return len(self.encs)
    def __getitem__(self, i):
        return {k: torch.tensor(v) for k, v in self.encs[i].items()}

def align_batch(tokenizer, tokens: List[str], bio_labels: List[str],
                label2id: Dict[str, int], max_length: int):
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=max_length)
    word_ids = enc.word_ids()
    labels = [-100] * len(enc["input_ids"])
    prev_w = None
    for i, w in enumerate(word_ids):
        if w is None:
            continue
        lab = bio_labels[w]
        if w != prev_w:  # label only first subword
            labels[i] = label2id.get(lab, label2id["O"]) if lab in label2id else label2id["O"]
        prev_w = w
    enc["labels"] = labels
    return enc

def build_splits(records: List[Dict], valid_frac: float = 0.1, seed: int = 42):
    # split by record (not by token)
    idx = np.arange(len(records))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - valid_frac))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    train = [records[i] for i in tr_idx]
    valid = [records[i] for i in va_idx]
    return train, valid

def compute_metrics_builder(id2label):
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids
        true_all, pred_all = [], []
        for y_true, y_pred in zip(labels, preds):
            yt, yp = [], []
            for t, p_ in zip(y_true, y_pred):
                if t == -100:
                    continue
                yt.append(id2label[int(t)])
                yp.append(id2label[int(p_)])
            true_all.append(yt)
            pred_all.append(yp)
        return {"f1_seqeval": f1_score(true_all, pred_all)}
    return compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--train_path", type=str, default="data/raw/Tagged_Titles_Train.tsv.gz")
    parser.add_argument("--labels_path", type=str, default="configs/labels.txt")
    parser.add_argument("--outdir", type=str, default="checkpoints/xlmr")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    # Load labels and data
    labels = load_label_list(args.labels_path)
    bio_labels, id2label, label2id = build_bio_maps(labels)

    df = read_tagged_train(args.train_path)
    records = to_bio_sequences(df)
    train_records, valid_records = build_splits(records, valid_frac=0.1, seed=cfg.get("seed", 42))

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("model_name", "xlm-roberta-base"))
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.get("model_name", "xlm-roberta-base"),
        id2label=id2label,
        label2id=label2id,
        num_labels=len(bio_labels),
    )

    # Encode datasets
    max_len = int(cfg.get("max_length", 160))
    train_enc = [align_batch(tokenizer, r["tokens"], r["bio_labels"], label2id, max_len) for r in train_records]
    valid_enc = [align_batch(tokenizer, r["tokens"], r["bio_labels"], label2id, max_len) for r in valid_records]

    train_ds = EncodedDataset(train_enc)
    valid_ds = EncodedDataset(valid_enc)

    args_hf = TrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=int(cfg.get("batch_size", 16)),
        per_device_eval_batch_size=int(cfg.get("batch_size", 16)),
        learning_rate=float(cfg.get("learning_rate", 3e-5)),
        num_train_epochs=int(cfg.get("epochs", 8)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        warmup_ratio=float(cfg.get("warmup_ratio", 0.06)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 1)),
        fp16=bool(cfg.get("fp16", True)),
        load_best_model_at_end=True,
        metric_for_best_model="f1_seqeval",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args_hf,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_builder(id2label),
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    print(f"\n⏱️ Training took {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")
    os.makedirs(args.outdir, exist_ok=True)
    trainer.save_model(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    print("Training complete. Model saved to:", args.outdir)

if __name__ == "__main__":
    main()
