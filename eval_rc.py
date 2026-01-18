# eval_rc.py
# Evaluate TF-IDF + OneVsRest(Logistic) root-cause model
# - Prints & saves per-class Precision/Recall/F1/AP
# - Saves PR curves and simple feature-importance bars
# Outputs:
#   outputs/metrics/rc_metrics.json
#   outputs/plots/rc_pr_<label>.png
#   outputs/plots/rc_features_<label>.png

import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

BASE = Path(".")
RC_DIR = BASE / "rc_outputs"
VAL_CSV = BASE / "data/processed/rc_val_manual.csv"
PLOTS = BASE / "outputs/plots"; PLOTS.mkdir(parents=True, exist_ok=True)
METR  = BASE / "outputs/metrics"; METR.mkdir(parents=True, exist_ok=True)

# ---------- Load artifacts ----------
vec = joblib.load(RC_DIR / "tfidf.pkl")
clf = joblib.load(RC_DIR / "ovr_logreg.pkl")

# labels.json may be a list or {"labels":[...]}
labels_all = json.load(open(RC_DIR / "labels.json", "r", encoding="utf-8"))
if isinstance(labels_all, dict) and "labels" in labels_all:
    labels_all = labels_all["labels"]

# Drop unwanted heads globally (your request)
DROP = {"overwhelmed"}

# Keep order from training, but remove DROPs and anything not present in CSV
df = pd.read_csv(VAL_CSV)
csv_heads = {c for c in df.columns if c != "text"}  # expected multi-label columns
keep_idx = [i for i, L in enumerate(labels_all) if (L not in DROP and L in csv_heads)]
labels   = [labels_all[i] for i in keep_idx]

if "text" not in df.columns:
    raise ValueError(f"'text' column missing in {VAL_CSV}. Found: {df.columns.tolist()}")
missing_in_csv = [L for L in labels if L not in df.columns]
if missing_in_csv:
    raise ValueError(f"These labels expected by model but missing in CSV: {missing_in_csv}")

# thresholds.json may be {"thresholds": {...}} or flat {...}
th = json.load(open(RC_DIR / "thresholds.json", "r", encoding="utf-8"))
thr_all = th["thresholds"] if isinstance(th, dict) and "thresholds" in th else th
thr = {L: float(thr_all.get(L, 0.5)) for L in labels}

# ---------- Build matrices ----------
X = vec.transform(df["text"].astype(str).values)
Y = df[labels].astype(int).values  # shape (n, k_kept)

# Model probabilities (one column per original head); then filter columns to `keep_idx`
if hasattr(clf, "predict_proba"):
    P_all = np.vstack([est.predict_proba(X)[:, 1] for est in clf.estimators_]).T
else:
    from scipy.special import expit
    P_all = np.vstack([expit(est.decision_function(X)) for est in clf.estimators_]).T

# Ensure shapes compatible
if P_all.shape[1] < max(keep_idx) + 1:
    raise ValueError(
        f"Model output has {P_all.shape[1]} heads but labels.json index needs {max(keep_idx)+1}."
    )
P = P_all[:, keep_idx]  # shape (n, k_kept)

# ---------- Metrics ----------
def binarize(P, tdict=None):
    out = np.zeros_like(P, dtype=int)
    for j, lbl in enumerate(labels):
        t = 0.5 if tdict is None else float(tdict.get(lbl, 0.5))
        out[:, j] = (P[:, j] >= t).astype(int)
    return out

pred_05  = binarize(P, None)
pred_thr = binarize(P, thr)

def micro_macro(Y, Yh):
    return {
        "micro/precision": float(precision_score(Y, Yh, average="micro", zero_division=0)),
        "micro/recall":    float(recall_score(Y, Yh, average="micro", zero_division=0)),
        "micro/f1":        float(f1_score(Y, Yh, average="micro", zero_division=0)),
        "macro/precision": float(precision_score(Y, Yh, average="macro", zero_division=0)),
        "macro/recall":    float(recall_score(Y, Yh, average="macro", zero_division=0)),
        "macro/f1":        float(f1_score(Y, Yh, average="macro", zero_division=0)),
    }

report_05  = classification_report(Y, pred_05,  target_names=labels, zero_division=0, output_dict=True)
report_thr = classification_report(Y, pred_thr, target_names=labels, zero_division=0, output_dict=True)

summary = {
    "labels": labels,
    "micro_macro@0.5": micro_macro(Y, pred_05),
    "micro_macro@thresholds": micro_macro(Y, pred_thr),
    "per_class@0.5": {
        lbl: {
            "precision": report_05[lbl]["precision"],
            "recall":    report_05[lbl]["recall"],
            "f1":        report_05[lbl]["f1-score"],
            "support":   int(report_05[lbl]["support"]),
            "AP":        float(average_precision_score(Y[:, i], P[:, i])),
        }
        for i, lbl in enumerate(labels)
    },
    "per_class@thresholds": {
        lbl: {
            "precision": report_thr[lbl]["precision"],
            "recall":    report_thr[lbl]["recall"],
            "f1":        report_thr[lbl]["f1-score"],
            "support":   int(report_thr[lbl]["support"]),
            # AP is threshold-independent, so we don’t repeat it here
        }
        for i, lbl in enumerate(labels)
    },
}
json.dump(summary, open(METR / "rc_metrics.json", "w"), indent=2)
print("Saved metrics to", METR / "rc_metrics.json")

# ---------- PR Curves ----------
for i, lbl in enumerate(labels):
    y, p = Y[:, i], P[:, i]
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    plt.figure()
    plt.step(rec, prec, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall — {lbl} (AP={ap:.3f})")
    plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
    plt.grid(True, alpha=0.3)
    out_png = PLOTS / f"rc_pr_{lbl}.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close()
print("Saved PR plots to", PLOTS)

# ---------- Feature importance (LogReg coefficients) ----------
try:
    vocab = {v: k for k, v in vec.vocabulary_.items()}
    coefs = [getattr(est, "coef_", None) for est in clf.estimators_]
    # coefs are for all heads; select the same kept indices
    coefs = [coefs[i] for i in keep_idx]

    for lbl, coef in zip(labels, coefs):
        if coef is None:
            continue
        w = coef.ravel()
        topk = np.argsort(-np.abs(w))[:15]
        names = [vocab.get(int(j), str(j)) for j in topk]
        vals = w[topk]
        order = np.argsort(np.abs(vals))
        names = [names[k] for k in order]
        vals = vals[order]

        plt.figure(figsize=(6, 4))
        plt.barh(range(len(names)), vals)
        plt.yticks(range(len(names)), names)
        plt.xlabel("Coefficient")
        plt.title(f"Top features — {lbl}")
        plt.tight_layout()
        plt.savefig(PLOTS / f"rc_features_{lbl}.png", dpi=160)
        plt.close()
    print("Saved feature bars to", PLOTS)
except Exception as e:
    print("Feature importance plot skipped:", e)
