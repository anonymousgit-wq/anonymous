# src/rl_core.py - FIXED RL stub
from config_loader import load_all_configs
from models import get_models
from typing import Dict, Any


def run_rl_tpo():
    """RL-TPO++: PPO on TPO trajectories (stub for now)"""
    configs = load_all_configs()
    rl_cfg = configs["rl_config.yaml"]
    
    # FIXED: safe config access
    algorithm = rl_cfg.get("algorithm", "ppo")
    epochs = rl_cfg.get("ppo_epochs", 4)
    
    print("RL-TPO++ Starting...")
    print(f"Algorithm: {algorithm}, Epochs: {epochs}")
    print("N trajectories:", rl_cfg.get("n_trajectories", 10000))
    print("LoRA rank:", rl_cfg.get("lora_rank", 16))
    print("✅ RL config loaded (PPO stub ready)")
    print("Next: generate TPO trajectories → PPO training")
    # TODO: full TRL/PPO implementation


if __name__ == "__main__":
    run_rl_tpo()
