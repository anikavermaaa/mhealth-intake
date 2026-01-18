# rc_infer.py
import os
import json
import numpy as np
import joblib

BASE = os.path.join(os.path.dirname(__file__), "rc_outputs")

# --- load artifacts ---
tfidf = joblib.load(os.path.join(BASE, "tfidf.pkl"))
ovr_logreg = joblib.load(os.path.join(BASE, "ovr_logreg.pkl"))

# thresholds.json may be either {"thresholds": {...}} or {...}
with open(os.path.join(BASE, "thresholds.json"), "r", encoding="utf-8") as f:
    _t = json.load(f)
    thresholds = _t["thresholds"] if isinstance(_t, dict) and "thresholds" in _t else _t

# labels.json may be a list or {"labels": [...]}
with open(os.path.join(BASE, "labels.json"), "r", encoding="utf-8") as f:
    _l = json.load(f)
    labels = _l["labels"] if isinstance(_l, dict) and "labels" in _l else _l

# --- drop labels you don't want to serve ---
DROP = {"overwhelmed"}
labels = [L for L in labels if L not in DROP]

def _predict_proba_ovr(clf, X):
    """
    Return P with shape (n_samples, n_classes) for OneVsRest-like classifiers.
    Works across sklearn versions (predict_proba / decision_function).
    """
    # Try vectorized predict_proba first
    if hasattr(clf, "predict_proba"):
        P = clf.predict_proba(X)
        if isinstance(P, list):  # some versions return list of per-class arrays
            P = np.column_stack(P)
        return P

    # Fallback: decision_function -> sigmoid
    if hasattr(clf, "decision_function"):
        from scipy.special import expit
        parts = []
        for est in clf.estimators_:
            if hasattr(est, "predict_proba"):
                parts.append(est.predict_proba(X)[:, 1])
            elif hasattr(est, "decision_function"):
                parts.append(expit(est.decision_function(X)))
            else:
                parts.append(est.predict(X).astype(float))
        return np.column_stack(parts)

    raise RuntimeError("Classifier has neither predict_proba nor decision_function.")

def predict_root_causes(text: str):
    """
    Returns:
      {
        'proba': {label: prob, ...},
        'hits' : [labels crossing threshold]
      }
    Notes:
      - Assumes the classifier was trained with label order matching labels.json.
      - Ignores any classifier.classes_ numeric encoding and maps by index.
    """
    X = tfidf.transform([str(text)])

    # (1, n_classes_in_model)
    P_all = _predict_proba_ovr(ovr_logreg, X)

    # If model has more/less classes than labels, trim to the min length
    n = min(len(labels), P_all.shape[1])
    if n != len(labels):
        # keep only the first n labels to avoid index errors
        active_labels = labels[:n]
    else:
        active_labels = labels

    # Map by index (training stored consistent order with labels.json)
    probs = {lbl: float(P_all[0, i]) for i, lbl in enumerate(active_labels)}

    # Apply thresholds; default to 0.5 if a label threshold is missing
    hits = [lbl for lbl, p in probs.items() if p >= thresholds.get(lbl, 0.5)]

    return {"proba": probs, "hits": hits}
