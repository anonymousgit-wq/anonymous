# eval_tpo.py (Fixed import + path check)
import json
import subprocess
import statistics
import time
import os  # ← ADDED
from typing import Callable

def ollama_generate(model_name: str, prompt: str, max_tokens: int = 256) -> str:
    cmd = ["ollama", "run", model_name, "--no-tty", prompt]
    try:
        start = time.time()
        result = subprocess.check_output(cmd, text=True, timeout=60, stderr=subprocess.DEVNULL)
        elapsed = time.time() - start
        print(f"  ⏱️  {model_name}: {elapsed:.1f}s")
        return result.strip().split('\n')[0]
    except Exception as e:
        print(f"  ❌ {model_name} error: {e}")
        return "Error"

def ollama_score(model_name: str, prompt: str, answer: str) -> float:
    score_prompt = f"Rate 1-10 (higher=better): Prompt: {prompt[:200]}... Response: {answer[:200]}...\nScore:"
    cmd = ["ollama", "run", model_name, "--no-tty", score_prompt]
    try:
        result = subprocess.check_output(cmd, text=True, timeout=30).strip()
        score = float([w for w in result.split() if w.replace('.','').isdigit()][0])
        return min(max(score, 1.0), 10.0)
    except:
        return 5.0

def eval_dataset(dataset_path: str, generator: Callable, scorer: Callable, n_examples: int = 20):
    if not os.path.exists(dataset_path):  # ← FIXED
        print(f"❌ Missing {dataset_path}")
        return 0, 0, 0
    
    scores = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f.readlines()[:n_examples]]
    
    print(f"\n🧪 {dataset_path} ({len(examples)} examples)")
    
    for i, ex in enumerate(examples):
        if i % 5 == 0: print(f"  {i}/{len(examples)}")
        answer = generator(ex["prompt"])
        score = scorer(ex["prompt"], answer)
        scores.append(score)
    
    return statistics.mean(scores), statistics.stdev(scores), len(scores)

# CONFIG - UPDATE YOUR MODEL NAMES
BASELINE_MODEL = "llama3.1:8b"
REWARD_MODEL = "qwen2.5:7b-instruct"

# RUN
datasets = ["alpacaeval_eval.jsonl", "arena_hard.jsonl", "hh_rlhf_eval.jsonl"]
results = {}

print("🚀 RL-TPO++ Benchmark")
print("=" * 40)

for dataset in datasets:
    base_mean, base_std, n = eval_dataset(
        dataset,
        lambda p: ollama_generate(BASELINE_MODEL, p),
        lambda p, a: ollama_score(REWARD_MODEL, p, a)
    )
    
    # TPO placeholder (replace with your TPO func later)
    tpo_mean, tpo_std, _ = eval_dataset(
        dataset,
        lambda p: ollama_generate(BASELINE_MODEL, p),  # TEMP: same as baseline
        lambda p, a: ollama_score(REWARD_MODEL, p, a)
    )
    
    delta = tpo_mean - base_mean
    results[dataset] = {
        "baseline": f"{base_mean:.2f}±{base_std:.2f}",
        "tpo": f"{tpo_mean:.2f}±{tpo_std:.2f}",
        "delta": f"{delta:+.2f}",
        "n": n
    }
    
    print(f"📊 {dataset}: Base={base_mean:.2f} TPO={tpo_mean:.2f} Δ={delta:+.2f}")

# SAVE
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Saved benchmark_results.json")
print("\n📋 Paper table:")
print(json.dumps(results, indent=2))
