# model_infer.py
import torch
from transformers import AutoTokenizer
from train_two_heads import TwoHeadBERT, clean_text  # relies on your training module

class TwoHeadInfer:
    def __init__(self, ckpt_path: str):
        """
        Load the trained two-head BERT from a checkpoint produced by train_two_heads.py.
        Expects the checkpoint dict to contain:
          - 'args': {'model_name': str, 'max_len': int, ...}
          - 'state_dict': model state dict
        """
        self.ckpt_path = ckpt_path
        ckpt = torch.load(ckpt_path, map_location="cpu")
        args = ckpt["args"]

        self.model_name = args["model_name"]
        self.max_len = int(args.get("max_len", 256))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TwoHeadBERT(self.model_name).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        # Debug telemetry for anxiety head weights/bias
        with torch.no_grad():
            w_mean = float(self.model.anx_head.weight.abs().mean().item())
            b_val = float(self.model.anx_head.bias.item())
        print(f"[TwoHeadInfer] loaded: {ckpt_path}")
        print(f"[TwoHeadInfer] anx_head |w|mean={w_mean:.4f} bias={b_val:.4f}")

    @torch.no_grad()
    def _encode(self, text: str):
        """Encode a single string to model inputs on the correct device."""
        text = clean_text(text or "")
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @torch.no_grad()
    def predict(self, text: str):
        """
        Predict depression/anxiety probabilities for a single text.
        Ensures eval-mode, no_grad, CPU-safe return.
        Returns:
            {'dep_prob': float, 'anx_prob': float}
        """
        # safety: make sure eval mode is set at call-time too
        self.model.eval()

        ids, mask = self._encode(text)

        # forward returns raw logits per head; apply sigmoid for independent binary heads
        dep_logit = self.model(ids, mask, head="dep")
        anx_logit = self.model(ids, mask, head="anx")

        dep_prob = torch.sigmoid(dep_logit).float().item()
        anx_prob = torch.sigmoid(anx_logit).float().item()

        # clamp to avoid any numeric edge cases, then round for UI
        dep_prob = float(max(1e-6, min(1.0 - 1e-6, dep_prob)))
        anx_prob = float(max(1e-6, min(1.0 - 1e-6, anx_prob)))

        return {
            "dep_prob": round(dep_prob, 3),
            "anx_prob": round(anx_prob, 3),
        }

    @torch.no_grad()
    def predict_batch(self, texts):
        """
        Optional: batched inference for a list of strings (keeps same API/heads).
        Returns list of dicts like predict().
        """
        self.model.eval()

        # tokenize as a batch
        cleaned = [clean_text(t or "") for t in texts]
        enc = self.tokenizer(
            cleaned, return_tensors="pt", truncation=True, padding=True, max_length=self.max_len
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        dep_logits = self.model(enc["input_ids"], enc["attention_mask"], head="dep").squeeze(-1)
        anx_logits = self.model(enc["input_ids"], enc["attention_mask"], head="anx").squeeze(-1)

        dep_probs = torch.sigmoid(dep_logits).float().cpu().numpy()
        anx_probs = torch.sigmoid(anx_logits).float().cpu().numpy()

        out = []
        for dp, ap in zip(dep_probs, anx_probs):
            dp = float(max(1e-6, min(1.0 - 1e-6, dp)))
            ap = float(max(1e-6, min(1.0 - 1e-6, ap)))
            out.append({"dep_prob": round(dp, 3), "anx_prob": round(ap, 3)})
        return out
