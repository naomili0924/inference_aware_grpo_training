# Inference-Aware Reinforcement Learning Training Framework

A lightweight extension of vLLM that exposes speculative decoding statistics during inference, enabling research on **inference-aware reinforcement learning** and **GRPO-style training objectives**.

The project replaces the default vLLM engine with a custom engine that tracks speculative decoding acceptance rates and other decoding efficiency metrics while maintaining compatibility with the standard `vllm.LLM` API.

---

## Motivation

Modern language model training optimizes for output quality but is largely unaware of inference efficiency.

Speculative decoding introduces a draft model that proposes tokens which are then verified by a larger target model. The efficiency of speculative decoding depends heavily on the draft model's acceptance rate.

This repository provides:

- A custom vLLM engine with speculative decoding telemetry
- Per-request acceptance rate statistics accessible after `generate()`
- A drop-in replacement for `vllm.LLM`
- A full GRPO training loop with an inference-aware composite reward
- GSM8K math training with exact answer correctness as the task score

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
step    6 | loss=-0.3406 | reward=-2.440 | accept=0.238 | latency=3511ms | kv=3.95MB
step    7 | loss=-0.3696 | reward=-2.279 | accept=0.243 | latency=3360ms | kv=3.85MB
step    8 | loss=-0.3083 | reward=-2.497 | accept=0.227 | latency=3555ms | kv=3.96MB
step    9 | loss=-0.3619 | reward=-2.511 | accept=0.230 | latency=3574ms | kv=3.95MB
step   10 | loss=-0.2132 | reward=-2.921 | accept=0.180 | latency=3933ms | kv=3.96MB
step   11 | loss=-0.4339 | reward=-2.575 | accept=0.219 | latency=3627ms | kv=3.94MB
step   12 | loss=-0.3880 | reward=-2.694 | accept=0.208 | latency=3734ms | kv=3.95MB
step   13 | loss=-0.3266 | reward=-2.558 | accept=0.216 | latency=3607ms | kv=3.94MB
step   14 | loss=-0.3186 | reward=-2.319 | accept=0.245 | latency=3397ms | kv=3.95MB
step   15 | loss=-0.1997 | reward=-2.250 | accept=0.254 | latency=3336ms | kv=3.98MB
step   16 | loss=-0.4058 | reward=-2.444 | accept=0.230 | latency=3507ms | kv=3.93MB
step   17 | loss=-0.3654 | reward=-2.481 | accept=0.229 | latency=3543ms | kv=3.95MB
step   18 | loss=-0.3160 | reward=-2.586 | accept=0.214 | latency=3632ms | kv=3.98MB
step   19 | loss=-0.3086 | reward=-2.436 | accept=0.229 | latency=3497ms | kv=3.94MB
step   20 | loss=-0.2634 | reward=-2.264 | accept=0.251 | latency=3347ms | kv=3.96MB
step   21 | loss=-0.2862 | reward=-2.396 | accept=0.232 | latency=3461ms | kv=3.95MB
step   22 | loss=-0.2707 | reward=-2.232 | accept=0.255 | latency=3319ms | kv=3.95MB
step   23 | loss=-0.3840 | reward=-2.792 | accept=0.197 | latency=3822ms | kv=3.94MB
step   24 | loss=-0.4479 | reward=-2.260 | accept=0.245 | latency=3345ms | kv=3.76MB
step   25 | loss=-0.3683 | reward=-2.472 | accept=0.229 | latency=3533ms | kv=3.96MB
step   26 | loss=-0.3320 | reward=-2.621 | accept=0.211 | latency=3665ms | kv=3.95MB
step   27 | loss=-0.3732 | reward=-2.458 | accept=0.232 | latency=3523ms | kv=3.97MB
step   28 | loss=-0.2288 | reward=-2.535 | accept=0.216 | latency=3583ms | kv=3.98MB
step   29 | loss=-0.3833 | reward=-2.420 | accept=0.233 | latency=3487ms | kv=3.93MB
step   30 | loss=-0.3193 | reward=-2.396 | accept=0.234 | latency=3462ms | kv=3.96MB
step   31 | loss=-0.1914 | reward=-2.263 | accept=0.257 | latency=3352ms | kv=3.99MB
step   32 | loss=-0.4257 | reward=-2.359 | accept=0.244 | latency=3436ms | kv=3.96MB
step   33 | loss=-0.3593 | reward=-2.183 | accept=0.266 | latency=3281ms | kv=3.95MB
step   34 | loss=-0.3128 | reward=-2.551 | accept=0.216 | latency=3599ms | kv=3.94MB
step   35 | loss=-0.3156 | reward=-2.316 | accept=0.244 | latency=3393ms | kv=3.95MB
step   36 | loss=-0.2513 | reward=-2.150 | accept=0.266 | latency=3248ms | kv=3.95MB
step   37 | loss=-0.2644 | reward=-2.398 | accept=0.233 | latency=3464ms | kv=3.98MB
step   38 | loss=-0.2797 | reward=-2.707 | accept=0.201 | latency=3741ms | kv=3.95MB
step   39 | loss=-0.4212 | reward=-2.271 | accept=0.253 | latency=3357ms | kv=3.95MB
step   40 | loss=-0.4327 | reward=-2.374 | accept=0.245 | latency=3451ms | kv=3.95MB
step   41 | loss=-0.4343 | reward=-2.351 | accept=0.242 | latency=3425ms | kv=3.92MB
step   42 | loss=-0.3123 | reward=-2.718 | accept=0.200 | latency=3751ms | kv=3.93MB
step   43 | loss=-0.2991 | reward=-2.687 | accept=0.198 | latency=3717ms | kv=3.95MB
step   44 | loss=-0.2374 | reward=-2.001 | accept=0.276 | latency=3115ms | kv=3.82MB
step   45 | loss=-0.4073 | reward=-2.345 | accept=0.245 | latency=3422ms | kv=3.94MB
step   46 | loss=-0.2606 | reward=-2.421 | accept=0.228 | latency=3485ms | kv=3.88MB
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

### Results — 50 Steps

Target: `Qwen/Qwen2.5-1.5B-Instruct` · Draft: `Qwen/Qwen2.5-0.5B-Instruct`  
500 training samples · batch=2 · G=2 rollouts · max_new_tokens=256 · lr=1e-6

| Metric | Value |
|--------|-------|
| Baseline accuracy (GSM8K test) | 69.0% |
| Final accuracy (GSM8K test) | **73.0%** |
| Spec accept rate (avg) | **60–70%** |

> Math reasoning produces significantly higher spec accept rates than general text (60–70% vs 20–30%), because chain-of-thought arithmetic follows structured, predictable patterns that the 0.5B draft model can anticipate.

### Eval progression (every 10 steps)

```
Baseline eval on 100 test problems...
  Baseline accuracy: 69.0%

step    1 | loss= 0.0094 | correct=0.56 | accept=0.653 | latency=3263ms | reward=-2.390
step    2 | loss=-0.0658 | correct=0.44 | accept=0.586 | latency=5033ms | reward=-4.474
...
step   10 | loss= 0.0022 | correct=0.88 | accept=0.653 | latency=3241ms | reward=-2.042
  >>> Eval accuracy after step 10: 70.0%
...
step   20 | loss=-0.0235 | correct=0.81 | accept=0.662 | latency=3410ms | reward=-2.284
  >>> Eval accuracy after step 20: 74.0%
...
step   30 | loss=-0.0194 | correct=0.25 | accept=0.593 | latency=4118ms | reward=-3.661
  >>> Eval accuracy after step 30: 70.0%
...
step   40 | loss= 0.0074 | correct=0.75 | accept=0.668 | latency=3687ms | reward=-2.656
  >>> Eval accuracy after step 40: 69.0%
...
step   50 | loss=-0.0571 | correct=0.69 | accept=0.642 | latency=4434ms | reward=-3.530
  >>> Eval accuracy after step 50: 70.0%

Final eval...
  Baseline: 69.0%  →  Final: 73.0%
```

### Usage

```bash
python playground/train_grpo_math.py
```

Key config knobs in `GRPOConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_train_samples` | 500 | Set to `None` for all 7,473 examples |
| `num_train_steps` | 50 | Training steps |
| `eval_every` | 10 | Eval on test set every N steps |
| `eval_samples` | 100 | Test problems per eval |
| `num_rollouts_per_prompt` | 4 | G in GRPO |
| `max_new_tokens` | 512 | Tokens for chain-of-thought |

---

## Repository Structure

```
inference_aware_grpo_training/
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
└── train_grpo_math.py          # GRPO on GSM8K with answer correctness reward
```

---

## Setup

```bash
git clone https://github.com/naomili0924/inference_aware_grpo_training.git
cd inference_aware_grpo_training
pip install -e .
```

Requires vLLM 0.19.0, PyTorch 2.10, and a CUDA GPU.
