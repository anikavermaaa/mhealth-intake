# make_report.py
import json, os
from pathlib import Path
import pandas as pd

BASE   = Path(".")
METR   = BASE / "outputs/metrics"
PLOTS  = BASE / "outputs/plots"
REPORT = BASE / "outputs/report"
REPORT.mkdir(parents=True, exist_ok=True)

# ---- load metrics ----
rc_path   = METR / "rc_metrics.json"
bert_path = METR / "bert_twoheads_metrics.json"

rc = json.load(open(rc_path, "r", encoding="utf-8"))
bt = json.load(open(bert_path, "r", encoding="utf-8"))

# ---- tables: RC micro/macro + per-class ----
rc_micro = pd.DataFrame([rc["micro_macro@0.5"], rc["micro_macro@thresholds"]], index=["RC@0.5","RC@thresholds"])
rc_pc_05 = pd.DataFrame(rc["per_class@0.5"]).T[["precision","recall","f1","support","AP"]]
rc_pc_th = pd.DataFrame(rc["per_class@thresholds"]).T[["precision","recall","f1","support"]]

# ---- tables: BERT heads ----
bert_rows = []
for name, d in bt.items():
    bert_rows.append({"head": name, "AP": d["AP"], "F1@0.5": d["F1@0.5"], "Precision@0.5": d["Precision@0.5"], "Recall@0.5": d["Recall@0.5"]})
bert_df = pd.DataFrame(bert_rows).set_index("head")

# ---- write Markdown report ----
md = []

md.append("# Model Evaluation Report")
md.append("")
md.append("## Root-Cause (TF-IDF + One-Vs-Rest Logistic)")
md.append("")
md.append("**Micro/Macro (two operating points):**")
md.append(rc_micro.round(3).to_markdown())
md.append("")
md.append("**Per-class @ 0.5 threshold:**")
md.append(rc_pc_05.round(3).to_markdown())
md.append("")
md.append("**Per-class @ tuned thresholds:**")
md.append(rc_pc_th.round(3).to_markdown())
md.append("")

# embed PR curves and feature bars
md.append("**PR Curves (RC):**")
for lbl in rc["labels"]:
    img = PLOTS / f"rc_pr_{lbl}.png"
    if img.exists():
        md.append(f"![PR {lbl}]({img.as_posix()})")
md.append("")
md.append("**Top Features (LogReg coefficients per class):**")
for lbl in rc["labels"]:
    img = PLOTS / f"rc_features_{lbl}.png"
    if img.exists():
        md.append(f"![Top features {lbl}]({img.as_posix()})")

md.append("")
md.append("## BERT Two-Head (Depression / Anxiety)")
md.append("")
md.append(bert_df.round(3).to_markdown())
md.append("")
for name in ["anxiety","depression"]:
    img = Path(bt[name]["plot"])
    if img.exists():
        md.append(f"![PR {name}]({img.as_posix()})")

md_text = "\n".join(md)
(md_out := REPORT / "evaluation_report.md").write_text(md_text, encoding="utf-8")

# ---- write a simple HTML too (GitHub-style table rendering) ----
html = [
    "<!doctype html><meta charset='utf-8'><title>Model Evaluation Report</title>",
    "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:980px;margin:40px auto;padding:0 16px;line-height:1.4} img{max-width:100%;height:auto} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px 10px} th{background:#f6f8fa} h1,h2{margin-top:28px}</style>",
    f"<h1>Model Evaluation Report</h1>",
    f"<h2>Root-Cause (TF-IDF + OVR)</h2>",
    "<h3>Micro/Macro</h3>",
    rc_micro.round(3).to_html(),
    "<h3>Per-class @ 0.5</h3>",
    rc_pc_05.round(3).to_html(),
    "<h3>Per-class @ tuned thresholds</h3>",
    rc_pc_th.round(3).to_html(),
    "<h3>PR Curves (RC)</h3>",
]
for lbl in rc["labels"]:
    img = PLOTS / f"rc_pr_{lbl}.png"
    if img.exists(): html.append(f"<p><img src='../plots/rc_pr_{lbl}.png' alt='PR {lbl}'></p>")
html.append("<h3>Top Features (RC)</h3>")
for lbl in rc["labels"]:
    img = PLOTS / f"rc_features_{lbl}.png"
    if img.exists(): html.append(f"<p><img src='../plots/rc_features_{lbl}.png' alt='Top features {lbl}'></p>")
html += [
    "<h2>BERT Two-Head</h2>",
    bert_df.round(3).to_html(),
    "<h3>PR Curves (BERT)</h3>",
]
for name in ["anxiety","depression"]:
    if Path(bt[name]["plot"]).exists():
        html.append(f"<p><img src='../plots/bert_pr_{name}.png' alt='PR {name}'></p>")

( REPORT / "evaluation_report.html" ).write_text("\n".join(html), encoding="utf-8")

print("Wrote:", md_out)
print("Wrote:", REPORT / "evaluation_report.html")
