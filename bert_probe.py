import sys, torch
from model_infer import TwoHeadInfer

ckpt = "outputs/twohead/best.pt"   # <-- change if your checkpoint lives elsewhere
infer = TwoHeadInfer(ckpt)
infer.model.eval()
infer.model.to("cpu")

tests = [
    "Okay, just checking the app.",
    "I feel fine and had a normal day.",
    "Heart is racing and I can't stop worrying.",
    "Feeling low and nothing seems enjoyable lately.",
]

for t in tests:
    try:
        probs = infer.predict(t)  # expect {'dep_prob': x, 'anx_prob': y}
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
    print(f"{t}\n  -> {probs}\n")
