import csv
from typing import List, Dict, Tuple
import pandas as pd

# === Reading helpers ===

def read_tagged_train(path: str) -> pd.DataFrame:
    """
    Read the human-tagged training data with strict NA handling.
    Columns: Record Number, Category Id, Title, Token, Tag
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=None,
        compression="gzip",
        quoting=csv.QUOTE_NONE,
        engine="python",
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def read_listings(path: str) -> pd.DataFrame:
    """
    Read the big listings file.
    Columns: Record Number, Category Id, Title
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=None,
        compression="gzip",
        quoting=csv.QUOTE_NONE,
        engine="python",
    )
    df.columns = [c.strip() for c in df.columns]
    return df


# === Token / Label processing ===

def whitespace_tokens(title: str) -> List[str]:
    """Split by whitespace only; preserve punctuation exactly (Annexure rule)."""
    if not isinstance(title, str):
        title = str(title)
    return [t for t in title.split() if t != ""]


def to_bio_sequences(tagged_df: pd.DataFrame) -> List[Dict]:
    """
    Convert continuation tags → BIO per record.
    Returns a list of {record, category, tokens, bio_labels} dicts.
    """
    records = []
    for rid, g in tagged_df.groupby("Record Number", sort=False):
        tokens = g["Token"].tolist()
        cats = g["Category"].unique().tolist()
        cat = cats[0] if cats else None
        raw_tags = g["Tag"].tolist()

        bio = []
        prev_tag = "O"
        span_open = False
        for tok, tag in zip(tokens, raw_tags):
            # Continuation: empty tag means same as previous non-empty
            if tag == "":
                tag = prev_tag
            if tag == "O":
                bio.append("O")
                span_open = False
            else:
                if (not span_open) or (tag != prev_tag):
                    bio.append(f"B-{tag}")
                    span_open = True
                else:
                    bio.append(f"I-{tag}")
            prev_tag = tag

        records.append(
            {
                "record": rid,
                "category": cat,
                "tokens": tokens,
                "bio_labels": bio,
            }
        )
    return records


def load_label_list(labels_txt_path: str) -> List[str]:
    labels = [
        l.strip()
        for l in open(labels_txt_path, "r", encoding="utf-8").read().splitlines()
        if l.strip()
    ]
    return labels


def build_bio_maps(labels: List[str]) -> Tuple[List[str], Dict[int, str], Dict[str, int]]:
    """Create BIO label list and id maps (O kept as is)."""
    bio_labels = ["O"] + [p + "-" + l for l in labels if l != "O" for p in ["B", "I"]]
    id2label = {i: lab for i, lab in enumerate(bio_labels)}
    label2id = {lab: i for i, lab in id2label.items()}
    return bio_labels, id2label, label2id


def bio_to_spans(tokens: List[str], bio_labels: List[str]) -> List[Tuple[str, str]]:
    """
    Convert BIO labels into (tag, value) spans by joining tokens with single space.
    Keeps duplicates as separate spans if they occur.
    """
    spans = []
    cur_tag, cur_tokens = None, []
    for tok, lab in zip(tokens, bio_labels):
        if lab == "O" or lab is None:
            if cur_tag:
                spans.append((cur_tag, " ".join(cur_tokens)))
                cur_tag, cur_tokens = None, []
            continue
        bi, tag = lab.split("-", 1)
        if bi == "B" or (cur_tag and tag != cur_tag):
            if cur_tag:
                spans.append((cur_tag, " ".join(cur_tokens)))
            cur_tag, cur_tokens = tag, [tok]
        else:  # I-*
            cur_tokens.append(tok)
    if cur_tag:
        spans.append((cur_tag, " ".join(cur_tokens)))
    return spans
