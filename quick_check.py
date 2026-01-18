import pandas as pd

for name in ["anxiety", "depression"]:
    df = pd.read_csv(f"data/processed/{name}_clean.csv")
    print(name.upper(), df.shape)
    print(df.head(2).to_dict(orient="records"))
