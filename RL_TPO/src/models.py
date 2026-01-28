# src/models.py - 5-role Ollama wrapper
import ollama
import yaml
from typing import Dict, Any, Optional
from config_loader import load_all_configs


class OllamaRole:
    def __init__(self, config: Dict[str, Any]):
        self.tag = config["tag"]
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.95)
        self.max_tokens = config.get("max_tokens", 512)
        self.role_desc = config.get("role", "")

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = ollama.chat(
            model=self.tag,
            messages=messages,
            options={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
        )
        return resp["message"]["content"].strip()


class TPO_Models:
    def __init__(self, project_root: str = r"D:\Research\RL_TPO"):
        configs = load_all_configs(project_root)
        models_cfg = configs["models.yaml"]["models"]
        
        self.policy = OllamaRole(models_cfg["policy"])
        self.rm_primary = OllamaRole(models_cfg["rm_primary"])
        self.loss_critic = OllamaRole(models_cfg["loss_critic"])
        self.gradient_gen = OllamaRole(models_cfg["gradient_gen"])
        self.consensus_rm = OllamaRole(models_cfg["consensus_rm"])
        
        print("✅ 5 TPO models loaded")


models = None  # global instance


def get_models() -> TPO_Models:
    global models
    if models is None:
        models = TPO_Models()
    return models
