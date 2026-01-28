# get_alpacaeval_data.py (ASCII-safe for Windows)
import json
import urllib.request
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

print("Downloading AlpacaEval from your HF link...")

url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval.json"

with urllib.request.urlopen(url) as response:
    data = json.loads(response.read().decode('utf-8'))

print(f"Loaded {len(data)} examples")

# Print ALL prompts summary
print("\nALL PROMPTS SUMMARY:")
print("=" * 80)
for i, ex in enumerate(data):
    instruction = ex["instruction"]
    input_text = ex.get("input", "")
    
    prompt_type = "instruction only" if not input_text else "instruction + input"
    print(f"{i+1:3d}. {prompt_type:<20} | {instruction[:80]}...")

print(f"\nGenerated {len(data)} prompts")

# Save JSONL
with open("alpacaeval_eval.jsonl", "w", encoding="utf-8") as f:
    for i, ex in enumerate(data):
        instruction = ex["instruction"]
        input_text = ex.get("input", "")
        
        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}\n\nResponse:"
        else:
            prompt = f"{instruction}\n\nResponse:"
        
        item = {
            "id": str(i),
            "prompt": prompt.strip(),
            "instruction": instruction,
            "input": input_text,
            "dataset": "alpaca_eval_2"
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Saved alpacaeval_eval.jsonl")
