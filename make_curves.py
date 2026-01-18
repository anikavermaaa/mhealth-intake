# make_curves.py
import json, numpy as np, pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# ---------- paths ----------
BASE  = Path(".")
PLOTS = BASE / "outputs/plots"; PLOTS.mkdir(parents=True, exist_ok=True)

# ---------- BERT (depression/anxiety) ----------
from model_infer import TwoHeadInfer

A_CSV = BASE / "data/processed/anxiety_test.csv"
D_CSV = BASE / "data/processed/depression_test.csv"

def load_bin(csv_path: Path):
    df = pd.read_csv(csv_path)
    text_col  = next((c for c in df.columns if c.lower() in {"text","statement","utterance","input"}), None)
    label_col = next((c for c in df.columns if c.lower() in {"label","target","y"}), None)
    if not text_col or not label_col:
        raise ValueError(f"{csv_path} must contain text/label columns. Found: {list(df.columns)}")
    return df[[text_col, label_col]].rename(columns={text_col:"text", label_col:"label"})

anx_df = load_bin(A_CSV)
dep_df = load_bin(D_CSV)

# load model once
ckpt_candidates = [
    "outputs/twohead/best.pt",
    "outputs/debug_run/best.pt",
    "outputs/best.pt",
]
ckpt = next((p for p in ckpt_candidates if Path(p).exists()), None)
if ckpt is None:
    raise FileNotFoundError("No two-head checkpoint found under outputs/…")
infer = TwoHeadInfer(ckpt)

def score_df_batched(df, head: str, batch_size=64):
    texts = df["text"].astype(str).tolist()
    y     = df["label"].astype(int).to_numpy()
    probs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        preds = infer.predict_batch(chunk)  # list of dicts
        probs.extend([p["anx_prob"] if head=="anx" else p["dep_prob"] for p in preds])
    return y, np.array(probs, dtype=float)

y_anx, p_anx = score_df_batched(anx_df, "anx")
y_dep, p_dep = score_df_batched(dep_df, "dep")

# ----- BERT: ROC (combined) -----
fpr_anx, tpr_anx, _ = roc_curve(y_anx, p_anx)
fpr_dep, tpr_dep, _ = roc_curve(y_dep, p_dep)
auc_anx = auc(fpr_anx, tpr_anx)
auc_dep = auc(fpr_dep, tpr_dep)

plt.figure(figsize=(6,5))
plt.plot(fpr_anx, tpr_anx, label=f"Anxiety (AUC={auc_anx:.3f})")
plt.plot(fpr_dep, tpr_dep, label=f"Depression (AUC={auc_dep:.3f})")
plt.plot([0,1], [0,1], linestyle="--", linewidth=1)  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC — BERT Two-Head (Combined)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "bert_roc_combined.png", dpi=160)
plt.close()

# ----- BERT: PR (combined) -----
prec_anx, rec_anx, _ = precision_recall_curve(y_anx, p_anx)
prec_dep, rec_dep, _ = precision_recall_curve(y_dep, p_dep)
ap_anx = average_precision_score(y_anx, p_anx)
ap_dep = average_precision_score(y_dep, p_dep)

plt.figure(figsize=(6,5))
plt.step(rec_anx, prec_anx, where="post", label=f"Anxiety (AP={ap_anx:.3f})")
plt.step(rec_dep, prec_dep, where="post", label=f"Depression (AP={ap_dep:.3f})")
# random baseline = positive class prevalence (per curve, show both)
base_anx = y_anx.mean(); base_dep = y_dep.mean()
plt.hlines(base_anx, 0, 1, linestyles="dashed", linewidth=1)
plt.hlines(base_dep, 0, 1, linestyles="dashed", linewidth=1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall — BERT Two-Head (Combined)")
plt.legend()
plt.ylim([0,1.05]); plt.xlim([0,1])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "bert_pr_combined.png", dpi=160)
plt.close()

# ---------- Root-Cause ML (TF-IDF + OVR Logistic, micro-averaged curves) ----------
import joblib, json
from sklearn.metrics import roc_auc_score

RC_DIR  = BASE / "rc_outputs"
VAL_CSV = BASE / "data/processed/rc_val_manual.csv"

vec = joblib.load(RC_DIR / "tfidf.pkl")
clf = joblib.load(RC_DIR / "ovr_logreg.pkl")

labels_all = json.load(open(RC_DIR / "labels.json", "r", encoding="utf-8"))
if isinstance(labels_all, dict) and "labels" in labels_all:
    labels_all = labels_all["labels"]
DROP = {"overwhelmed"}  # you chose to drop this across the project

df_rc = pd.read_csv(VAL_CSV)
if "text" not in df_rc.columns:
    raise ValueError(f"'text' column missing in {VAL_CSV}. Found: {df_rc.columns.tolist()}")

csv_heads = [c for c in df_rc.columns if c != "text"]
keep_idx  = [i for i,L in enumerate(labels_all) if (L not in DROP and L in csv_heads)]
labels    = [labels_all[i] for i in keep_idx]

X = vec.transform(df_rc["text"].astype(str).values)
Y = df_rc[labels].astype(int).values   # shape (n, k)

# model probabilities for all heads then slice to kept
if hasattr(clf, "predict_proba"):
    P_all = np.vstack([est.predict_proba(X)[:,1] for est in clf.estimators_]).T
else:
    from scipy.special import expit
    P_all = np.vstack([expit(est.decision_function(X)) for est in clf.estimators_]).T
P = P_all[:, keep_idx]  # shape (n, k)

# ----- Micro-averaged ROC -----
fpr_rc, tpr_rc, _ = roc_curve(Y.ravel(), P.ravel())
auc_rc = auc(fpr_rc, tpr_rc)

plt.figure(figsize=(6,5))
plt.plot(fpr_rc, tpr_rc, label=f"Root-Cause (micro) AUC={auc_rc:.3f}")
plt.plot([0,1], [0,1], linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC — Root-Cause ML (micro-averaged)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "rc_roc_micro.png", dpi=160)
plt.close()

# ----- Micro-averaged PR -----
prec_rc, rec_rc, _ = precision_recall_curve(Y.ravel(), P.ravel())
ap_rc = average_precision_score(Y.ravel(), P.ravel())

plt.figure(figsize=(6,5))
plt.step(rec_rc, prec_rc, where="post", label=f"Root-Cause (micro) AP={ap_rc:.3f}")
# baseline at overall positive rate:
pos_rate = Y.mean()
plt.hlines(pos_rate, 0, 1, linestyles="dashed", linewidth=1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall — Root-Cause ML (micro-averaged)")
plt.legend()
plt.ylim([0,1.05]); plt.xlim([0,1])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "rc_pr_micro.png", dpi=160)
plt.close()

# ----- Print a summary to console -----
summary = {
    "bert": {
        "ROC_AUC": {"anxiety": float(auc_anx), "depression": float(auc_dep)},
        "PR_AP"  : {"anxiety": float(ap_anx),  "depression": float(ap_dep)}
    },
    "rc_ml": {
        "ROC_AUC_micro": float(auc_rc),
        "PR_AP_micro"  : float(ap_rc)
    }
}
print(json.dumps(summary, indent=2))
print("Saved plots to", str(PLOTS))
