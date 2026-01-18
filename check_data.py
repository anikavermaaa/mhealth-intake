import pandas as pd

# Read the Excel files from data/raw
anx = pd.read_excel("data/raw/anxiety.xlsx")
dep = pd.read_excel("data/raw/depression.xlsx")

print("\nANXIETY DATA")
print("Shape:", anx.shape)
print("Columns:", anx.columns.tolist())
print("Unique labels:", anx["is_stressed/anxious"].unique())
print("Missing Text:", anx["Text"].isna().sum())

print("\nDEPRESSION DATA")
print("Shape:", dep.shape)
print("Columns:", dep.columns.tolist())
print("Unique labels:", dep["is_depression"].unique())
print("Missing clean_text:", dep["clean_text"].isna().sum())
