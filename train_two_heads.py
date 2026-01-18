import os, json, re, random, argparse, warnings
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ------------------ Utils ------------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def class_report(series: pd.Series) -> Dict:
    counts = series.value_counts(dropna=False).to_dict()
    total = len(series)
    pct = {k: round(v * 100.0 / total, 2) for k, v in counts.items()}
    return {"counts": counts, "percent": pct}

# ------------------ Data -------------------
class BinaryTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding=False, max_length=self.max_len)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "label": torch.tensor(self.labels[i], dtype=torch.float),
        }

def collate_pad(batch, pad_id: int):
    # dynamic padding
    ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]
    y = torch.stack([b["label"] for b in batch])
    maxlen = max(x.size(0) for x in ids)
    ids_pad = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    attn_pad = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, (ii, aa) in enumerate(zip(ids, attn)):
        ids_pad[i, : ii.size(0)] = ii
        attn_pad[i, : aa.size(0)] = aa
    return {"input_ids": ids_pad, "attention_mask": attn_pad, "labels": y}

# --------------- Model ---------------------
class TwoHeadBERT(nn.Module):
    def __init__(self, base_model="bert-base-uncased", dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.dep_head = nn.Linear(hidden, 1)
        self.anx_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, head: str):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        x = self.drop(cls)
        if head == "dep":
            return self.dep_head(x).squeeze(-1)
        elif head == "anx":
            return self.anx_head(x).squeeze(-1)
        else:
            raise ValueError("head must be 'dep' or 'anx'")

# --------------- Metrics -------------------
def bin_metrics_from_logits(logits: np.ndarray, labels: np.ndarray, thresh=0.5):
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= thresh).astype(int)
    acc = float((preds == labels).mean()) if len(labels) else 0.0
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

@torch.no_grad()
def eval_head(model, device, loader, head: str):
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["labels"].cpu().numpy().astype(int)
        logits = model(ids, am, head=head).cpu().numpy()
        logits_all.append(logits); labels_all.append(y)
    if not logits_all:
        return {"acc":0.0,"precision":0.0,"recall":0.0,"f1":0.0,"tp":0,"fp":0,"fn":0}
    logits = np.concatenate(logits_all); labels = np.concatenate(labels_all)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in bin_metrics_from_logits(logits, labels).items()}

# --------- Train loop (alternating heads) ---
def run_epoch_alt(model, device, dep_loader, anx_loader, optimizer, scheduler, loss_fn, train=True):
    if train: model.train()
    else: model.eval()

    dep_iter = iter(dep_loader) if dep_loader is not None else None
    anx_iter = iter(anx_loader) if anx_loader is not None else None
    steps = max(len(dep_loader) if dep_loader else 0, len(anx_loader) if anx_loader else 0)
    total_loss = 0.0

    for _ in range(steps):
        for head, it in [("dep", dep_iter), ("anx", anx_iter)]:
            if it is None: continue
            try:
                batch = next(it)
            except StopIteration:
                continue
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            logits = model(ids, am, head=head)
            loss = loss_fn(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler: scheduler.step()
            total_loss += loss.item()
    denom = (len(dep_loader) if dep_loader else 0) + (len(anx_loader) if anx_loader else 0)
    return total_loss / max(1, denom)

# -------------------- Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dep_train", default="data/processed/depression_train.csv")
    ap.add_argument("--dep_val",   default="data/processed/depression_val.csv")
    ap.add_argument("--anx_train", default="data/processed/anxiety_train.csv")
    ap.add_argument("--anx_val",   default="data/processed/anxiety_val.csv")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="outputs/twohead")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)

    # Load data
    dep_tr = pd.read_csv(args.dep_train)
    dep_va = pd.read_csv(args.dep_val)
    anx_tr = pd.read_csv(args.anx_train)
    anx_va = pd.read_csv(args.anx_val)

    # Clean (defensive)
    for df in [dep_tr, dep_va, anx_tr, anx_va]:
        df["text"] = df["text"].astype(str).map(clean_text)
        df.dropna(subset=["text","label"], inplace=True)
        df["label"] = df["label"].astype(int)

    # Reports
    dep_rep = class_report(pd.concat([dep_tr["label"], dep_va["label"]], ignore_index=True))
    anx_rep = class_report(pd.concat([anx_tr["label"], anx_va["label"]], ignore_index=True))
    print("Depression label report:", dep_rep)
    print("Anxiety label report:   ", anx_rep)
    if len(anx_rep["counts"]) < 2:
        warnings.warn("Anxiety set appears single-class; the anxiety head cannot learn a boundary without negatives.")

    # Tokenizer + datasets
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    def mk(ds): return BinaryTextDataset(ds["text"].tolist(), ds["label"].tolist(), tok, args.max_len)

    dep_train_ds, dep_val_ds = mk(dep_tr), mk(dep_va)
    anx_train_ds, anx_val_ds = mk(anx_tr), mk(anx_va)

    dep_train_loader = DataLoader(dep_train_ds, batch_size=args.batch_size, sampler=RandomSampler(dep_train_ds),
                                  collate_fn=lambda b: collate_pad(b, tok.pad_token_id))
    dep_val_loader   = DataLoader(dep_val_ds,   batch_size=args.batch_size, sampler=SequentialSampler(dep_val_ds),
                                  collate_fn=lambda b: collate_pad(b, tok.pad_token_id))
    anx_train_loader = DataLoader(anx_train_ds, batch_size=args.batch_size, sampler=RandomSampler(anx_train_ds),
                                  collate_fn=lambda b: collate_pad(b, tok.pad_token_id))
    anx_val_loader   = DataLoader(anx_val_ds,   batch_size=args.batch_size, sampler=SequentialSampler(anx_val_ds),
                                  collate_fn=lambda b: collate_pad(b, tok.pad_token_id))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoHeadBERT(args.model_name, dropout=0.1).to(device)

    # Optimizer + scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],  "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr)

    total_steps = (len(dep_train_loader) + len(anx_train_loader)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = nn.BCEWithLogitsLoss()

    best_avg_f1 = -1.0
    history = []

    print("Training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch_alt(model, device, dep_train_loader, anx_train_loader, optimizer, scheduler, loss_fn, train=True)
        dep_val_metrics = eval_head(model, device, dep_val_loader, head="dep")
        anx_val_metrics = eval_head(model, device, anx_val_loader, head="anx")
        avg_f1 = (dep_val_metrics["f1"] + anx_val_metrics["f1"]) / 2.0

        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "dep": dep_val_metrics,
            "anx": anx_val_metrics,
            "avg_f1": round(avg_f1, 4),
        }
        history.append(log)
        print(json.dumps(log, indent=2))

        # save best
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "best_log": log},
                       os.path.join(args.out_dir, "best.pt"))
            with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"best": log, "history": history}, f, indent=2)

    # final save
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "history": history},
               os.path.join(args.out_dir, "last.pt"))
    with open(os.path.join(args.out_dir, "metrics_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
