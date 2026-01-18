# eval_bert_twoheads.py
import os, json, re, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
from model_infer import TwoHeadInfer

BASE = Path(".")
PLOTS = BASE/"outputs/plots"; PLOTS.mkdir(parents=True, exist_ok=True)
METR  = BASE/"outputs/metrics"; METR.mkdir(parents=True, exist_ok=True)

# ---- where to read test data (adjust if needed) ----
A_CSV = BASE/"data/processed/anxiety_test.csv"
D_CSV = BASE/"data/processed/depression_test.csv"

# We expect columns: text and label (0/1). Try to guess safely.
def load_bin(csv_path):
    df = pd.read_csv(csv_path)
    # find text col
    text_col = next((c for c in df.columns if c.lower() in {"text", "statement", "input", "utterance"}), None)
    if text_col is None: raise ValueError(f"No text column found in {csv_path}")
    # find label col
    label_col = next((c for c in df.columns if c.lower() in {"label","target","y"}), None)
    if label_col is None: raise ValueError(f"No label column found in {csv_path}")
    return df[[text_col, label_col]].rename(columns={text_col:"text", label_col:"label"})

anx_df = load_bin(A_CSV)
dep_df = load_bin(D_CSV)

# ---- load model once ----
ckpt_candidates = [
    "outputs/twohead/best.pt",
    "outputs/debug_run/best.pt",
    "outputs/best.pt",
]
ckpt = next((p for p in ckpt_candidates if Path(p).exists()), None)
if ckpt is None: raise FileNotFoundError("No two-head checkpoint found.")
infer = TwoHeadInfer(ckpt)

def score_df(df, head="anx"):
    probs, y = [], []
    for t, lab in zip(df["text"].astype(str).tolist(), df["label"].astype(int).tolist()):
        out = infer.predict(t)
        y.append(int(lab))
        probs.append(float(out["anx_prob" if head=="anx" else "dep_prob"]))
    y = np.array(y); probs = np.array(probs)
    return y, probs

# ---- compute metrics + PR plots ----
def report_and_plot(y, p, name):
    ap = float(average_precision_score(y, p))
    # default 0.5 threshold for F1/P/R
    yhat = (p >= 0.5).astype(int)
    f1  = float(f1_score(y, yhat, zero_division=0))
    pr  = float(precision_score(y, yhat, zero_division=0))
    rc  = float(recall_score(y, yhat, zero_division=0))

    # PR curve
    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure()
    plt.step(rec, prec, where="post")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR Curve — {name} (AP={ap:.3f})")
    plt.ylim([0,1.05]); plt.xlim([0,1])
    plt.grid(True, alpha=0.3)
    out_png = PLOTS/f"bert_pr_{name}.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close()

    return {"AP": ap, "F1@0.5": f1, "Precision@0.5": pr, "Recall@0.5": rc, "plot": str(out_png)}

y_anx, p_anx = score_df(anx_df, head="anx")
y_dep, p_dep = score_df(dep_df, head="dep")

res = {
    "anxiety": report_and_plot(y_anx, p_anx, "anxiety"),
    "depression": report_and_plot(y_dep, p_dep, "depression"),
}
json.dump(res, open(METR/"bert_twoheads_metrics.json","w"), indent=2)
print("Saved metrics to", METR/"bert_twoheads_metrics.json")
print(json.dumps(res, indent=2))
