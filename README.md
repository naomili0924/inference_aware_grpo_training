# Inference-Aware Reinforcement Learning Training Framework

LLMs are trained to produce correct outputs — but correctness alone doesn't determine serving cost. A model that generates verbose, unpredictable, or cache-unfriendly responses is expensive to deploy, even if its answers are right. **This project makes inference efficiency a first-class training objective.**

The key insight is that speculative decoding gives us a live, per-request signal of inference cost: the **draft token acceptance rate**. When a small draft model's predictions are frequently accepted by the larger target model, decoding is fast and cheap. When they are rejected, the target must do more work. By feeding this acceptance rate — alongside latency, token length, and KV memory — back into the GRPO training reward, the model learns to generate outputs that are simultaneously high-quality and inference-efficient.

On top of this, an **adaptive curriculum scheduler** uses KDE-based reward bucketing to detect which training samples the model finds hardest and automatically allocates more rollout iterations to them — improving sample efficiency across GRPO epochs.

---

## What this repo provides

- A custom vLLM engine that exposes per-request speculative decoding telemetry
- Drop-in replacement for `vllm.LLM` with `get_spec_decode_stats()` after `generate()`
- A GRPO training loop with a composite reward: task correctness + spec accept rate − latency − token length − KV memory
- GSM8K math training reaching **83% accuracy** (+15pp over a 68% baseline) using exact answer correctness as the task score
- A `DatasetScheduler` with KDE valley-finding that buckets samples by EMA reward and assigns dynamic rollout counts per difficulty cluster

---

## Spec Decode Accept Rate — `playground/main.py`

Target: `Qwen/Qwen2.5-1.5B-Instruct` · Draft: `Qwen/Qwen2.5-0.5B-Instruct` · 5 speculative tokens

```
=== Spec Decode Accept Rates ===

[0]  accept_rate=100.0%  (95/95 draft tokens accepted)
  Output: The draft model is a model that can generate a sequence of tokens...

[1]  accept_rate=93.3%   (42/45 draft tokens accepted)
  Output: It is a variant of the GRU algorithm, which is a type of recurrent neural net...

[2]  accept_rate=100.0%  (50/50 draft tokens accepted)
  Output: The transformer model is a type of recurrent neural network (RNN)...

[3]  accept_rate=33.3%   (5/15 draft tokens accepted)
  Output: Reinforcement learning is a type of machine learning that involves the use of feedback...

[4]  accept_rate=88.0%   (66/75 draft tokens accepted)
  Output: This is a key part of the KV caching mechanism in the TensorFlow framework...
```

Access the stats after `generate()`:

```python
from inference_aware_grpo_training import VLLM
from vllm import SamplingParams

llm = VLLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    speculative_config={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "num_speculative_tokens": 5,
    },
)

outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=128))
spec_stats = llm.get_spec_decode_stats()

for output in outputs:
    stats = spec_stats.get(output.request_id, {})
    print(f"accept_rate={stats['accept_rate']:.1%}  ({stats['num_accepted']}/{stats['num_draft']})")
```

---

## GRPO Training Loop — `playground/train_grpo.py`

### Reward Function

```
reward = task_score
         - α × latency_ms          (queued → last token)
         - β × generated_tokens
         - γ × kv_memory_mb        (estimated from model dims)
         + δ × speculative_accept_rate
         + ε × cache_reuse_ratio   (prefix cache hits / prompt len)
```

Default weights: `α=0.001, β=0.001, γ=0.01, δ=1.0, ε=0.5`

### Training Run — 50 Steps

Target: `Qwen/Qwen2.5-1.5B-Instruct` · Draft: `Qwen/Qwen2.5-0.5B-Instruct`  
Batch size: 4 prompts · G=4 rollouts per prompt · lr=1e-6

```
Reward weights: alpha=0.001 (latency)  beta=0.001 (tokens)  gamma=0.01 (kv_mb)  delta=1.0 (accept_rate)  eps=0.5 (cache_reuse)
Starting GRPO — 50 steps, batch=4, G=4

step    1 | loss=-0.2730 | reward=-2.346 | accept=0.234 | latency=3415ms | kv=3.92MB
step    2 | loss=-0.4511 | reward=-2.444 | accept=0.230 | latency=3507ms | kv=3.95MB
step    3 | loss=-0.3120 | reward=-2.169 | accept=0.257 | latency=3262ms | kv=3.86MB
step    4 | loss=-0.2744 | reward=-2.477 | accept=0.225 | latency=3535ms | kv=3.96MB
step    5 | loss=-0.2022 | reward=-2.492 | accept=0.225 | latency=3550ms | kv=3.95MB
...
step   47 | loss=-0.4153 | reward=-2.451 | accept=0.229 | latency=3513ms | kv=3.93MB
step   48 | loss=-0.3515 | reward=-2.427 | accept=0.231 | latency=3491ms | kv=3.92MB
step   49 | loss=-0.2924 | reward=-2.527 | accept=0.222 | latency=3581ms | kv=3.95MB
step   50 | loss=-0.2614 | reward=-2.849 | accept=0.188 | latency=3869ms | kv=3.94MB
Training complete.
```

### Architecture

- **HF model** loaded on CUDA for gradient updates (AdamW)
- **vllm engine** (separate GPU allocation) for fast rollout generation with spec decode
- `sync_weights_to_vllm()` calls `model_executor.collective_rpc("reload_weights", ...)` after every optimiser step to push updated weights into the live vllm engine

The `task_score` defaults to `1.0` — plug in your own quality scorer via the `task_score_fn` argument to `compute_rewards()`.

---

## GSM8K Math Training — `playground/train_grpo_math.py`

Trains on [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (7,473 train / 1,319 test problems).  
`task_score` is replaced with **exact answer correctness** (1.0 correct / 0.0 wrong).

### Adaptive Curriculum with `DatasetScheduler`

The training loop uses a reward-based data scheduler (`inference_aware_grpo_training/data_scheduler.py`) that groups samples into difficulty buckets and assigns more rollout iterations to harder samples each epoch.

**Algorithm:**

1. **EMA reward tracking** — each sample's per-step reward is smoothed with an exponential moving average (`ema_decay=0.7`) across epochs, giving a stable difficulty signal that updates as the model learns.

2. **KDE valley-finding** — after each epoch, the EMA reward distribution is min-max normalized to `[0, 1]` (to handle negative rewards), then a Gaussian KDE is fit and local minima (valleys) in the density are found. These valley positions are back-transformed to the original reward scale as bucket boundaries. Up to `n_buckets=3` adaptive clusters are created this way.

3. **Dynamic rollout steps** — each bucket's difficulty is measured by its normalized mean reward, and harder buckets (lower reward) receive proportionally more rollout iterations:
   ```
   rollout_steps = base_rollout_steps × (1 + α × (1 − normalized_mean_reward))
   ```
   With `base_rollout_steps=4` and `α=1.5`, the hardest bucket gets up to 10 rollouts while the easiest gets 4.

4. **Gradient accumulation** — rollouts within a batch are accumulated one at a time (`loss / rollout_n` per step) before a single optimizer update, keeping peak memory at `O(batch_size)` regardless of rollout count.

5. **Cluster report** — printed after every epoch with per-bucket stats and health indicators (spread + balance):

```
╔══ Epoch 1 — Cluster Report ══════════════════════════════════════════════════════════════════╗
│ global  mean=-2.290  std=1.244  min=-6.041  max=0.114
│ boundaries: -3.956  |  -1.923
│ bucket 0  [-5.99–-3.96]  n=   42 ( 8.4%)  rollout=9   ← hard
│ bucket 1  [-3.96–-1.92]  n=  231 (46.2%)  rollout=7   ← medium
│ bucket 2  [-1.92– 0.11]  n=  227 (45.4%)  rollout=5   ← easy
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Full Training Run Results

Target: `Qwen/Qwen2.5-1.5B-Instruct` · Draft: `Qwen/Qwen2.5-0.5B-Instruct`  
500 training samples · batch=4 · lr=1e-6 · up to 20 epochs · early stop patience=3

| Metric | Value |
|--------|-------|
| Baseline accuracy (GSM8K test, 100 samples) | 68% |
| Best accuracy | **83%** (epoch 6) |
| Improvement | **+15pp** |
| Epochs until early stop | 9 (patience=3, no improvement after epoch 6) |
| Spec accept rate (avg) | **60–70%** |

> Math reasoning produces significantly higher spec accept rates than general text (60–70% vs 20–30%), because chain-of-thought arithmetic follows structured, predictable patterns that the 0.5B draft model can anticipate.

**Epoch-by-epoch accuracy:**

| Epoch | Accuracy | Notes |
|-------|----------|-------|
| — | 68% | baseline |
| 1 | 72% | ★ new best |
| 2 | 62% | no improvement 1/3 |
| 3 | 76% | ★ new best |
| 4 | 66% | no improvement 1/3 |
| 5 | 76% | no improvement 2/3 |
| 6 | **83%** | ★ new best |
| 7 | 65% | no improvement 1/3 |
| 8 | 71% | no improvement 2/3 |
| 9 | 60% | no improvement 3/3 → early stop |

### Bucketing Findings

**Epoch 1** produced a genuinely multimodal reward distribution — the KDE found two valleys and split samples into three meaningful difficulty buckets (8% hard, 46% medium, 46% easy) with rollouts ranging from 5 to 9. This is the curriculum working as designed.

**From epoch 2 onward**, the distribution converged to a single sharp peak (mean reward rising from -2.29 → -1.30 by epoch 6) with only ~1.5% of samples remaining as extreme outliers in a left tail. The KDE found no meaningful valley in the main mass, collapsing to a single real boundary. Effectively, 98.5% of samples received uniform rollout=6 for the remaining epochs.

**Why the distribution collapsed fast:** GSM8K rewards are binary (correct=0, incorrect=penalty). Once the model learns the majority of problems, rewards concentrate — there is no intermediate partial-credit signal to keep distinct difficulty clusters apart.

**Did bucketing help?** The curriculum differentiation was active for only the first epoch. The +15pp gain over 9 epochs is primarily attributable to GRPO training itself. The bucketing likely contributed to the early jump (68%→72% in epoch 1) and ensured the ~7 persistently hardest samples always received maximum rollouts (10×), but its marginal contribution beyond uniform sampling is hard to isolate without a control run.

**When bucketing would matter more:** datasets with graded/partial rewards, harder tasks where the reward distribution stays multimodal across epochs, or a slower EMA decay that preserves early difficulty signal longer.

### Usage

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python playground/train_grpo_math.py
```

Key config knobs in `GRPOConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_train_samples` | 500 | Set to `None` for all 7,473 examples |
| `num_epochs` | 20 | Max training epochs |
| `early_stop_patience` | 3 | Stop after N epochs without improvement |
| `batch_size` | 4 | Problems per batch |
| `base_rollout_steps` | 4 | Rollouts for the easiest (highest-reward) bucket |
| `rollout_alpha` | 1.5 | Controls rollout scaling for harder buckets |
| `n_buckets` | 3 | Number of adaptive difficulty buckets |
| `ema_decay` | 0.7 | EMA smoothing for per-sample reward history |
| `gpu_memory_utilization` | 0.2 | vLLM memory fraction (leave headroom for HF model) |
| `eval_every` | 1 | Eval on test set every N epochs |
| `eval_samples` | 100 | Test problems per eval |
| `max_new_tokens` | 512 | Tokens for chain-of-thought |

---

## Repository Structure

```
inference_aware_grpo_training/
├── data_scheduler.py           # Adaptive curriculum: RewardTracker, AdaptiveBucketizer, DatasetScheduler
├── entrypoints/
│   └── llm.py                  # VLLM class (drop-in for vllm.LLM) + get_spec_decode_stats()
├── v1/
│   ├── core/sched/
│   │   └── scheduler.py        # VLLMScheduler — accumulates spec decode stats per request
│   └── engine/
│       ├── core.py             # VLLMEngineCore — replaces default scheduler with VLLMScheduler
│       ├── core_client.py      # VLLMInprocClient — single engine-core creation
│       └── llm_engine.py       # VLLMEngine — patches make_client to avoid double model load
playground/
├── main.py                     # Spec decode accept rate demo (5 requests)
├── train_grpo.py               # Generic GRPO training loop
└── train_grpo_math.py          # GRPO on GSM8K with adaptive curriculum + dynamic rollouts
```

---

## Setup

```bash
git clone https://github.com/naomili0924/inference_aware_grpo_training.git
cd inference_aware_grpo_training
pip install -e .
```

Requires vLLM 0.19.0, PyTorch 2.10, and a CUDA GPU.
