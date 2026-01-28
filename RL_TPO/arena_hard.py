# arena_hard.py (Fixed - Handles real JSONL format)
import json
import urllib.request

print("📥 Downloading Arena-Hard from your HF link...")

url = "https://huggingface.co/datasets/lmarena-ai/arena-hard-auto/resolve/c75cc2581ae7ac4b66ce5c804c002fe5a7c6339e/data/arena-hard-v0.1/question.jsonl"

with urllib.request.urlopen(url) as response:
    lines = response.read().decode('utf-8').splitlines()
    data = []
    for line in lines:
        try:
            ex = json.loads(line.strip())
            data.append(ex)
        except:
            continue  # Skip corrupt lines

print(f"✅ Loaded {len(data)} raw examples")

# Robust field extraction
examples = []
for i, ex in enumerate(data[:200]):  # Top 200
    # Try multiple possible prompt fields
    prompt = (ex.get("question") or 
              ex.get("prompt") or 
              ex.get("instruction") or 
              str(ex.get("text", ""))).strip()
    
    if len(prompt) > 10:  # Valid prompt
        item = {
            "id": str(i),
            "prompt": prompt,
            "dataset": "arena_hard_v0.1",
            "raw": ex
        }
        examples.append(item)

with open("arena_hard.jsonl", "w", encoding="utf-8") as f:
    for item in examples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(examples)} valid examples → arena_hard.jsonl")
print("\n📊 First example:")
print(json.dumps(examples[0] if examples else {}, indent=2))
