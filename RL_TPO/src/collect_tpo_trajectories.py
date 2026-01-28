import os
import json
from typing import List, Dict, Tuple

from tqdm import tqdm
from tpo_core import TPO_Engine


def load_prompts_from_txt(path: str) -> List[str]:
    """Load one prompt per line from a plain text file."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def collect_tpo_trajectories(
    engine: TPO_Engine,
    prompts: List[str],
    output_path: str,
    overwrite: bool = False,
) -> None:
    """
    Run TPO on a list of prompts and save (query, response, reward) records
    to a JSONL file. Each line is a self-contained JSON object.

    Format per line:
    {
      "query": "user prompt",
      "response": "best TPO response",
      "reward": float  # final reward score from reward model
    }
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        ensure_dir(out_dir)

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Set overwrite=True to replace."
        )

    total = len(prompts)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for q in tqdm(prompts, desc="Collecting TPO trajectories", total=total):
            try:
                best_resp, best_score = engine.run_tpo(q)
                record = {
                    "query": q,
                    "response": best_resp,
                    "reward": float(best_score),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                # Skip problematic queries but keep going
                print(f"[WARN] Error on prompt: {q[:80]}... -> {e}")


def main():
    # 1. Initialize your TPO engine (uses your existing config/models)
    engine = TPO_Engine()  # adapt if you require config paths/args

    # 2. Load prompts
    prompts_file = os.path.join("data", "prompts.txt")
    prompts = load_prompts_from_txt(prompts_file)

    # 3. Output path
    out_path = os.path.join("data", "tpo_trajectories.jsonl")

    # 4. Collect trajectories
    collect_tpo_trajectories(
        engine=engine,
        prompts=prompts,
        output_path=out_path,
        overwrite=True,
    )

    print(f"✅ Saved {len(prompts)} trajectories to {out_path}")


if __name__ == "__main__":
    main()
