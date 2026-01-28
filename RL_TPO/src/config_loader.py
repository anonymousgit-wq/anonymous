# src/config_loader.py - Loads all 3 YAMLs
import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_all_configs(project_root: str = r"D:\Research\RL_TPO") -> Dict[str, Dict[str, Any]]:
    """Load models.yaml, tpo_config.yaml, rl_config.yaml"""
    configs = {}
    
    config_path = Path(project_root) / "config"
    
    for yaml_file in ["models.yaml", "tpo_config.yaml", "rl_config.yaml"]:
        with open(config_path / yaml_file) as f:
            configs[yaml_file] = yaml.safe_load(f)
    
    return configs


if __name__ == "__main__":
    configs = load_all_configs()
    print("✅ All configs loaded:")
    for name, cfg in configs.items():
        print(f"  {name}: {list(cfg.keys())}")
