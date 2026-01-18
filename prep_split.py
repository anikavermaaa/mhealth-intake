import pandas as pd
from sklearn.model_selection import train_test_split
import os, json

os.makedirs("data/processed", exist_ok=True)

def split_save(df, name, seed=42, test_size=0.1, val_size=0.1):
    # first split off test
    strat = df["label"] if df["label"].nunique() > 1 else None
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strat
    )

    # then split train and val
    strat2 = train_val["label"] if train_val["label"].nunique() > 1 else None
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=seed, stratify=strat2
    )

    # save files
    for split_name, d in [("train", train), ("val", val), ("test", test)]:
        out = f"data/processed/{name}_{split_name}.csv"
        d.to_csv(out, index=False)

    # make a summary report
    report = {
        "total": len(df),
        "splits": {k: len(v) for k, v in [("train", train), ("val", val), ("test", test)]},
        "labels": {
            "total": df["label"].value_counts().to_dict(),
            "train": train["label"].value_counts().to_dict(),
            "val": val["label"].value_counts().to_dict(),
            "test": test["label"].value_counts().to_dict(),
        }
    }
    with open(f"data/processed/{name}_split_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"{name} split report saved:", report["splits"])

# Anxiety (may be all 1s; stratification will skip automatically)
anx = pd.read_csv("data/processed/anxiety_clean.csv")
split_save(anx, "anxiety")

# Depression (has both 0 and 1)
dep = pd.read_csv("data/processed/depression_clean.csv")
split_save(dep, "depression")
