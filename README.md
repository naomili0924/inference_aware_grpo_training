# Inference-Aware GRPO Training

A lightweight extension of vLLM that exposes speculative decoding statistics during inference, enabling research on **inference-aware reinforcement learning** and **GRPO-style training objectives**.

The project replaces the default vLLM engine with a custom engine that tracks speculative decoding acceptance rates and other decoding efficiency metrics while maintaining compatibility with the standard `vllm.LLM` API.

---

## Motivation

Modern language model training optimizes for output quality but is largely unaware of inference efficiency.

Speculative decoding introduces a draft model that proposes tokens which are then verified by a larger target model. The efficiency of speculative decoding depends heavily on the draft model's acceptance rate.

This repository provides:

- A custom vLLM engine
- Speculative decoding telemetry
- Per-request acceptance statistics
- A drop-in replacement for `vllm.LLM`
- Infrastructure for future inference-aware GRPO/RL training experiments

The long-term goal is to explore training objectives that jointly optimize:

- Response quality
- Reward model score
- Inference latency
- Speculative decoding efficiency

---

## Features

### Drop-in vLLM Replacement

```python
from inference_aware_grpo_training import VLLM
