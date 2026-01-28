# src/main.py - Master runner
import argparse
from tpo_core import TPO_Engine
from rl_core import run_rl_tpo


def main():
    parser = argparse.ArgumentParser(description="RL-TPO++")
    parser.add_argument("--mode", choices=["tpo", "rl", "eval"], default="tpo")
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    if args.mode == "tpo":
        if not args.query:
            print("Error: --query required for tpo mode")
            return
        engine = TPO_Engine()
        resp, score = engine.run_tpo(args.query)
        print(resp)
        print(f"Score: {score}")

    elif args.mode == "rl":
        run_rl_tpo()

    elif args.mode == "eval":
        print("Eval mode: run benchmarks (TODO)")


if __name__ == "__main__":
    main()
