# prep_rc_split.py
import os, json, re
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

os.makedirs("data/processed", exist_ok=True)

# === CONFIG ===
RC_CSV = "data/processed/rootcause_clean.csv"   # <-- put your non-disorder dataset path here
TEXT_COL = "text"                               # <-- the text column
LABELS   = ["perfectionism","fear_of_failure","lack_of_interest","environment_distraction","dopamine_addiction"]  # edit to match your columns
SEED     = 42
SPLITS   = (0.8, 0.1, 0.1)  # train, val, test

def norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    df = pd.read_csv(RC_CSV)
    # basic checks
    for c in [TEXT_COL] + LABELS:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # normalize + dedup on text
    df["_norm"] = df[TEXT_COL].astype(str).map(norm_text)
    df = df.drop_duplicates(subset=["_norm"]).reset_index(drop=True)

    # drop labels that have zero positives
    keep = [c for c in LABELS if df[c].sum() > 0]
    dropped = [c for c in LABELS if c not in keep]
    if not keep:
        raise ValueError("All labels have zero positives after dedup. Check your data.")
    if dropped:
        print("Dropping zero-positive labels:", dropped)

    Y = df[keep].astype(int).values
    X_idx = np.arange(len(df))

    # train/val/test via multilabel stratified shuffles
    train_p, val_p, test_p = SPLITS
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(1-train_p), random_state=SEED)
    train_idx, hold_idx = next(msss1.split(X_idx, Y))

    Y_hold = Y[hold_idx]
    hold_ratio = test_p / (val_p + test_p)
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=hold_ratio, random_state=SEED)
    val_idx_rel, test_idx_rel = next(msss2.split(hold_idx, Y_hold))

    val_idx  = hold_idx[val_idx_rel]
    test_idx = hold_idx[test_idx_rel]

    splits = {
        "train": df.iloc[train_idx].drop(columns=["_norm"]),
        "val":   df.iloc[val_idx].drop(columns=["_norm"]),
        "test":  df.iloc[test_idx].drop(columns=["_norm"]),
    }

    # save
    for name, d in splits.items():
        d.to_csv(f"data/processed/rc_{name}.csv", index=False)

    # report
    def counts(d):
        return {c: int(d[c].sum()) for c in keep}
    report = {
        "total_rows": int(len(df)),
        "labels_used": keep,
        "labels_dropped": dropped,
        "split_sizes": {k: int(len(v)) for k,v in splits.items()},
        "per_label_counts": {
            "train": counts(splits["train"]),
            "val": counts(splits["val"]),
            "test": counts(splits["test"]),
        }
    }
    with open("data/processed/rc_split_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved splits and report at data/processed/")

if __name__ == "__main__":
    main()
