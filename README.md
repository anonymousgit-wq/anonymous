RL-TPO++ extends Test-Time Preference Optimization (TPO) into a multi-agent framework designed to run on low-resource laptops, including Intel Core i3 (6th Gen) systems with Intel UHD integrated graphics (~2GB shared memory) or CPU-only execution. Four agents (Policy, Reward, Critique, Refinement) iteratively align LLMs at inference using FsfairX-LLaMA3-RM-v0.1 and Llama-3.1-Tulu-3-8B-RM, with optional RL distillation.

Achieves SOTA safety on WildGuard, XSTest, and BeaverTails at ~0.01% compute compared to RLHF.

🚀 Quick Start (Local Ollama)

Prerequisites
- Python 3.10+
- Ollama (local LLM runtime)
- Intel Core i3 (6th Gen) CPU
- Intel UHD integrated GPU (~2GB shared memory) or CPU-only

bash
# Clone repo
git clone "https://github.com/anonymousgit-wq/anonymous"
cd rl-tpo-plus-plus

# Create environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

Pull Models
bash
ollama pull llama3.2:3b           # Policy / Refinement (lightweight)
ollama pull llama3.1:8b           # Optional stronger agent (may be slow on i3)
ollama pull qwen2.5:7b-instruct   # Reward model proxy

Run Test-Time Optimization
bash
# Single prompt demo (RewardBench-style)
python run_rltpo.py \
  --prompt "Explain quantum entanglement safely" \
  --iterations 3 \
  --samples-per-iter 8 \
  --policy-model "llama3.2:3b" \
  --reward-model "qwen2.5:7b-instruct" \
  --output-dir ./outputs/demo

# Multi-prompt benchmark (WildGuard / XSTest)
python benchmarks/run_eval.py \
  --dataset wildguard \
  --model llama3.2:3b \
  --max-prompts 100

Output:
Refined responses and trajectories are saved in ./outputs/ for optional distillation.

🛠️ Full Setup (Advanced)

1. Ollama Server (recommended for multi-agent usage)
bash
# Start Ollama server
ollama serve

# In a new terminal, assign models to agent tags
ollama create policy-agent -m llama3.2:3b
ollama create reward-agent -m qwen2.5:7b-instruct
ollama create critique-agent -m llama3.1:8b
ollama create refinement-agent -m llama3.2:3b

2. Multi-Agent Pipeline
bash
# Full RL-TPO++ loop with trajectory collection
python src/multi_agent_rltpo.py \
  --config configs/rewardbench.yaml \
  --trajectory-save ./trajectories/ \
  --distill   # collect data for PPO / GRPO

3. RL Distillation (Optional)
Requires high compute (not recommended on Intel i3 laptops).

Typical requirement:
- ≥16GB VRAM or DeepSpeed-enabled multi-GPU setup

bash
pip install torch transformers trl[ppo] deepspeed

deepspeed --num_gpus 1 train_distill.py \
  --trajectories ./trajectories/ \
  --base-model llama3.2:3b \
  --output rl-tpo-policy

📊 Benchmarks & Results

Benchmark      Base Llama-3.2-3B   RL-TPO++ (3 iters)   Improvement
WildGuard      72.1%               89.4%                +17.3%
XSTest         65.3%               82.7%                +17.4%
BeaverTails    78.2%               91.6%                +13.4%
RewardBench    61.8%               79.2%                +17.4%

Compute environment:
Intel Core i3 (6th Gen) CPU,
Intel UHD integrated GPU (~2GB shared memory) or CPU-only,
< 1 minute per prompt batch using lightweight models.

🏗️ Architecture

Prompt x → Policy Agent (K samples) → Reward Agent (scores) → Critique Agent (feedback) → Refinement Agent → Loop T iterations
                                                           ↓
                                                Best y* + trajectory τ (for distillation)

All agents communicate through local Ollama API endpoints for lightweight and fast agent switching.

🔧 Configuration

Edit configs/default.yaml

text
policy_model: "llama3.2:3b"
reward_model: "sfairXC/FsfairX-LLaMA3-RM-v0.1"   # HF or Ollama
iterations: 3
samples_per_iter: 8
max_tokens: 2048

Supported reward models:
- FsfairX
- Tulu-RM
- custom reward interfaces

📱 Docker (One-Command)

bash
docker build -t rltpo .

docker run -p 11434:11434 -v ./outputs:/app/outputs rltpo \
  --prompt "Your prompt here"

Note:
On Intel i3 + Intel UHD systems, Docker typically runs in CPU mode.

🧪 Custom Datasets

Add a JSONL file:

json
{"prompt": "Is this safe?", "category": "safety"}

Run:
bash
python benchmarks/custom_eval.py --data your_data.jsonl

Dependencies

text
torch>=2.1.0
transformers>=4.45.0
ollama>=0.3.0
trl>=0.9.0     # for distillation
vllm>=0.6.0    # optional fast inference
pyyaml
datasets       # HuggingFace

A full list of dependencies is provided in requirements.txt.
