import os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, f1_score

VEC_PATH = "rc_outputs/tfidf.pkl"
CLF_PATH = "rc_outputs/ovr_logreg.pkl"
LAB_PATH = "rc_outputs/labels.json"
VAL_CSV  = "data/processed/rc_val_manual.csv"
OUT_PATH = "rc_outputs/thresholds.json"

def load_labels(path):
    labels = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(labels, dict) and "labels" in labels:
        labels = labels["labels"]
    return labels

def predict_proba_ovr(clf, X):
    # Works for LogisticRegression or SGDClassifier wrapped as OneVsRest
    if hasattr(clf, "predict_proba"):
        return np.vstack([est.predict_proba(X)[:, 1] for est in clf.estimators_]).T
    else:
        from scipy.special import expit
        return np.vstack([expit(est.decision_function(X)) for est in clf.estimators_]).T

def main():
    vec = joblib.load(VEC_PATH)
    clf = joblib.load(CLF_PATH)
    labels = load_labels(LAB_PATH)

    # --- drop unwanted labels globally ---
    DROP = {"overwhelmed"}
    labels = [L for L in labels if L not in DROP]
    print("Using labels:", labels)

    df = pd.read_csv(VAL_CSV)
    text_col = "text"
    for c in [text_col] + labels:
        if c not in df.columns:
            raise ValueError(f"Missing column in {VAL_CSV}: {c}")

    X = vec.transform(df[text_col].astype(str).values)
    Y = df[labels].astype(int).values
    P = predict_proba_ovr(clf, X)

    new_thr, per_class_f1 = {}, {}
    for j, label in enumerate(labels):
        y_true = Y[:, j]
        y_prob = P[:, j]

        if y_true.sum() == 0:
            new_thr[label] = 0.5
            per_class_f1[label] = None
            continue

        p, r, t = precision_recall_curve(y_true, y_prob)
        if len(t) == 0:
            new_thr[label] = 0.5
            per_class_f1[label] = None
            continue

        f1 = (2 * p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-9)
        best_idx = int(f1.argmax())
        thr = float(t[best_idx])

        new_thr[label] = thr
        y_pred = (y_prob >= thr).astype(int)
        per_class_f1[label] = float(f1_score(y_true, y_pred))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump({"thresholds": new_thr, "val_f1": per_class_f1}, open(OUT_PATH, "w"), indent=2)
    print("\n✅ Wrote new thresholds to", OUT_PATH)
    print("Per-class F1:", json.dumps(per_class_f1, indent=2))

if __name__ == "__main__":
    main()
