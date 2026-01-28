# src/tpo_core.py - FIXED (KeyError fix)
from typing import List, Tuple, Dict, Any
from pathlib import Path

from models import get_models
from config_loader import load_all_configs


class TPO_Engine:
    def __init__(self, project_root: str = r"D:\Research\RL_TPO"):
        self.configs = load_all_configs(project_root)
        self.tpo_cfg = self.configs["tpo_config.yaml"]
        self.models = get_models()
        
        # FIXED: safer config access with defaults
        self.n_samples = self.tpo_cfg.get("n_samples", 5)
        self.n_steps = self.tpo_cfg.get("n_steps", 2)
        self.max_cache_size = self.tpo_cfg.get("max_cache_size", 50)

    def sample_candidates(self, query: str) -> List[str]:
        system = "You are a helpful, honest assistant. Answer clearly and safely."
        responses = []
        for _ in range(self.n_samples):
            responses.append(self.models.policy.generate(query, system))
        return responses

    def score_responses(self, query: str, responses: List[str]) -> List[float]:
        scores = []
        for resp in responses:
            prompt = (
                "Score this answer from -10 to 10 (helpfulness, safety, correctness):\n\n"
                f"Query: {query}\n\nResponse: {resp}\n\n"
                "Reply ONLY with a number."
            )
            out = self.models.rm_primary.generate(prompt)
            try:
                score = float(out.split()[0])
            except:
                score = 0.0
            scores.append(score)
        return scores

    def compute_textual_loss(self, query: str, chosen: str, rejected: str) -> str:
        prompt = (
            f"Query: {query}\n\nChosen (better): {chosen}\n\nRejected (worse): {rejected}\n\n"
            "Explain why chosen > rejected. Suggest improvements for chosen."
        )
        return self.models.loss_critic.generate(prompt)

    def compute_textual_gradient(self, loss_text: str) -> str:
        prompt = f"Critique: {loss_text}\n\n3-6 bullet instructions to improve."
        return self.models.gradient_gen.generate(prompt)

    def update_responses(self, query: str, gradient: str) -> List[str]:
        system = "Improve your answer using these instructions."
        prompt = f"Query: {query}\n\nInstructions: {gradient}\n\nImproved answer:"
        responses = []
        for _ in range(self.n_samples):
            responses.append(self.models.policy.generate(prompt, system))
        return responses

    def run_tpo(self, query: str) -> Tuple[str, float]:
        print(f"Running TPO: N={self.n_samples}, D={self.n_steps}")
        cache: List[Tuple[str, float]] = []
        
        # Initial sampling
        print("Step 0: Initial sampling...")
        responses = self.sample_candidates(query)
        scores = self.score_responses(query, responses)
        cache.extend(list(zip(responses, scores)))
        print(f"Initial avg score: {sum(s for _,s in cache)/len(cache):.2f}")

        # TPO iterations
        for step in range(self.n_steps):
            print(f"\n--- Iteration {step+1}/{self.n_steps} ---")
            cache.sort(key=lambda x: x[1])
            if len(cache) > self.max_cache_size:
                cache = cache[-self.max_cache_size:]

            rejected, r_score = cache[0]
            chosen, c_score = cache[-1]
            print(f"Chosen score: {c_score:.2f}, Rejected: {r_score:.2f}")

            loss_text = self.compute_textual_loss(query, chosen, rejected)
            grad_text = self.compute_textual_gradient(loss_text)
            
            print("Textual gradient preview:", grad_text[:100] + "...")
            
            new_responses = self.update_responses(query, grad_text)
            new_scores = self.score_responses(query, new_responses)
            cache.extend(list(zip(new_responses, new_scores)))

        # Best response
        cache.sort(key=lambda x: x[1])
        best_resp, best_score = cache[-1]
        print(f"\n✅ TPO complete. Final avg score: {sum(s for _,s in cache)/len(cache):.2f}")
        return best_resp, best_score


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    engine = TPO_Engine()
    best_resp, best_score = engine.run_tpo(args.query)
    
    print("\n" + "="*50)
    print("FINAL TPO ANSWER:")
    print(best_resp)
    print(f"\nFINAL SCORE: {best_score:.3f}")
    print("="*50)


if __name__ == "__main__":
    main()
