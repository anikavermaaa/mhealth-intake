import pandas as pd
import numpy as np

def show_stats(path, name):
    df = pd.read_csv(path)
    print(f"\n{name.upper()} rows: {len(df)}  cols: {df.columns.tolist()}")
    print("label counts:", df["label"].value_counts().to_dict())
    # rough token-ish length = word count
    lens = df["text"].astype(str).str.split().apply(len)
    pct = {p:int(np.percentile(lens, p)) for p in [50, 75, 90, 95, 99]}
    print("length percentiles (words):", pct)

show_stats("data/processed/anxiety_clean.csv", "anxiety")
show_stats("data/processed/depression_clean.csv", "depression")
