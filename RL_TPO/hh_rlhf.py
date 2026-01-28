# hh_rlhf.py (Fixed - Correct HH-RLHF fields)
from datasets import load_dataset
import json

print("📥 Downloading HH-RLHF...")

ds = load_dataset("Anthropic/hh-rlhf", split="train")  # helpful-base split

print(f"✅ Loaded {len(ds)} examples")
print("📦 Taking first 1000 for eval...")

with open("hh_rlhf_eval.jsonl", "w", encoding="utf-8") as f:
    for i in range(1000):
        ex = ds[i]
        
        # HH-RLHF uses 'chosen'/'rejected' format, extract prompt from first
        prompt = ex["chosen"].split("Assistant:")[0].strip() if "Assistant:" in ex["chosen"] else ex["chosen"][:500]
        
        item = {
            "id": str(i),
            "prompt": prompt,
            "chosen": ex["chosen"][:200] + "..." if len(ex["chosen"]) > 200 else ex["chosen"],
            "rejected": ex["rejected"][:200] + "..." if len(ex["rejected"]) > 200 else ex["rejected"],
            "dataset": "hh_rlhf_helpful_base"
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ Saved hh_rlhf_eval.jsonl")
print("\n📊 First example:")
print(json.dumps(item, indent=2))
