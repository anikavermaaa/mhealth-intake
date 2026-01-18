import pandas as pd
import random

anx = pd.read_excel("data/raw/anxiety.xlsx")

neutral_samples = [
    "I enjoyed my lunch today and went for a walk.",
    "Reading a book this evening felt relaxing.",
    "Had a normal day at work and watched a movie.",
    "Cooking dinner was fun and calm.",
    "I went shopping and enjoyed the nice weather.",
]

negatives = pd.DataFrame({
    "Text": [random.choice(neutral_samples) for _ in range(len(anx)//1)],
    "is_stressed/anxious": [0 for _ in range(len(anx)//1)]
})

anx_balanced = pd.concat([anx, negatives], ignore_index=True)
anx_balanced = anx_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

anx_balanced.to_excel("data/raw/anxiety_with_negatives.xlsx", index=False)
print("Saved data/raw/anxiety_with_negatives.xlsx with added 0-label rows.")
