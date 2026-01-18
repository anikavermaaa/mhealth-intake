import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

def clean_one(src_path, text_col, label_col, dest_path):
    df = pd.read_excel(src_path)  # reads .xlsx
    # Rename to a standard schema
    df = df.rename(columns={text_col: "text", label_col: "label"})

    # Ensure strings, trim, drop empties
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    # Drop exact duplicate texts
    df = df.drop_duplicates(subset=["text"])

    # Force labels into {0,1}
    df = df[pd.to_numeric(df["label"], errors="coerce").isin([0, 1])]
    df["label"] = df["label"].astype(int)

    # Save
    df.to_csv(dest_path, index=False, encoding="utf-8")
    print(f"Saved {dest_path}  -> rows={len(df)}  label_counts={df['label'].value_counts().to_dict()}")

# Anxiety
clean_one(
    src_path="data/raw/anxiety.xlsx",
    text_col="Text",
    label_col="is_stressed/anxious",
    dest_path="data/processed/anxiety_clean.csv",
)

# Depression
clean_one(
    src_path="data/raw/depression.xlsx",
    text_col="clean_text",
    label_col="is_depression",
    dest_path="data/processed/depression_clean.csv",
)
