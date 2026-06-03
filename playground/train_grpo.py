"""
GRPO training with an inference-aware composite reward signal.

Reward function (per request):
    reward = task_score
             - alpha  * latency_ms
             - beta   * generated_tokens
             - gamma  * kv_memory_mb
             + delta  * speculative_accept_rate
             + eps_w  * cache_reuse_ratio

where:
    task_score            quality of the generated output  (user-supplied callable,
                          defaults to 1.0 so the loop works out of the box)
    latency_ms            wall-clock time from request arrival to last token
    generated_tokens      number of tokens produced
    kv_memory_mb          KV-cache memory consumed by this request (estimated)
    speculative_accept_rate  fraction of draft tokens accepted by the target model
    cache_reuse_ratio     fraction of prompt tokens served from prefix cache

All raw metrics are z-score normalised across the batch before weighting
so that differently-scaled terms are balanced.

Algorithm (per step):
    1. Sample a batch of prompts.
    2. Generate G rollouts per prompt using spec decode (vllm).
    3. Compute the composite reward for every rollout.
    4. advantage_i = (reward_i - mean) / (std + eps)   [group-relative]
    5. GRPO loss = -E[ advantage_i * log p_theta(y | x) ]
    6. Gradient step on the HF target model.
    7. Sync updated weights back into the vllm engine.
"""

import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import SamplingParams

from inference_aware_grpo_training import VLLM


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class RewardWeights:
    alpha: float = 1e-3    # latency penalty      (per ms)
    beta:  float = 1e-3    # length penalty       (per token)
    gamma: float = 1e-2    # KV memory penalty    (per MB)
    delta: float = 1.0     # spec accept bonus
    eps_w: float = 0.5     # prefix cache bonus


@dataclass
class GRPOConfig:
    target_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_model:  str = "Qwen/Qwen2.5-0.5B-Instruct"

    # rollout
    num_rollouts_per_prompt: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.8

    # training
    learning_rate: float = 1e-6
    num_train_steps: int = 50
    batch_size: int = 4
    grad_clip: float = 1.0
    weight_sync_every: int = 1

    # vllm
    gpu_memory_utilization: float = 0.4
    max_model_len: int = 1024
    num_speculative_tokens: int = 5

    reward: RewardWeights = field(default_factory=RewardWeights)
    norm_eps: float = 1e-8   # advantage normalisation


TRAIN_PROMPTS = [
    "Summarize: Speculative decoding uses a fast draft model to propose tokens.",
    "Summarize: GRPO is a reinforcement learning algorithm for training language models.",
    "Summarize: The transformer uses self-attention to model long-range dependencies.",
    "Summarize: Reinforcement learning from human feedback aligns models with preferences.",
    "Summarize: KV caching stores key-value tensors to speed up autoregressive decoding.",
    "Summarize: Low-rank adaptation fine-tunes large models with few trainable parameters.",
    "Summarize: Flash attention reduces memory usage by computing attention in tiles.",
    "Summarize: Beam search explores multiple token sequences to find high-probability outputs.",
]


# ─── Reward helpers ───────────────────────────────────────────────────────────

def _kv_memory_mb(num_tokens: int, model_cfg) -> float:
    """Estimate KV-cache memory (MB) consumed by a single request."""
    hc = model_cfg.hf_config
    num_layers   = hc.num_hidden_layers
    num_kv_heads = hc.num_key_value_heads
    head_dim     = hc.hidden_size // hc.num_attention_heads
    dtype_bytes  = 2  # bfloat16
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    return num_tokens * bytes_per_token / (1024 ** 2)


def compute_rewards(
    outputs,
    spec_stats: dict,
    llm: VLLM,
    w: RewardWeights,
    task_score_fn: Optional[Callable] = None,
) -> List[float]:
    """
    Compute the composite reward for a list of RequestOutput objects.

    Args:
        outputs:        list of vllm RequestOutput
        spec_stats:     dict returned by llm.get_spec_decode_stats()
        llm:            the VLLM engine (used to query model config)
        w:              RewardWeights
        task_score_fn:  optional callable(prompt, response) -> float in [0, 1].
                        Defaults to 1.0 for every request.
    """
    model_cfg = llm.llm_engine.model_config
    rewards = []

    for out in outputs:
        req_id = out.request_id
        response = out.outputs[0].text
        prompt   = out.prompt

        # task quality
        task_score = task_score_fn(prompt, response) if task_score_fn else 1.0

        # latency: queued → last token (both are perf_counter, seconds → ms)
        m = out.metrics
        latency_ms = (m.last_token_ts - m.queued_ts) * 1e3 if m else 0.0

        # generated tokens
        gen_tokens = len(out.outputs[0].token_ids)

        # KV memory (prompt + generated)
        total_tokens = len(out.prompt_token_ids) + gen_tokens
        kv_mb = _kv_memory_mb(total_tokens, model_cfg)

        # speculative accept rate
        stats = spec_stats.get(req_id, {})
        accept_rate = stats.get("accept_rate", 0.0)

        # prefix cache reuse ratio
        prompt_len = len(out.prompt_token_ids)
        cached = getattr(out, "num_cached_tokens", 0) or 0
        cache_reuse = cached / prompt_len if prompt_len > 0 else 0.0

        reward = (
            task_score
            - w.alpha * latency_ms
            - w.beta  * gen_tokens
            - w.gamma * kv_mb
            + w.delta * accept_rate
            + w.eps_w * cache_reuse
        )
        rewards.append(reward)

    return rewards


def _znorm(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    """Z-score normalise a 1-D tensor."""
    return (tensor - tensor.mean()) / (tensor.std() + eps)


# ─── Weight sync ──────────────────────────────────────────────────────────────

def sync_weights_to_vllm(hf_model: torch.nn.Module, llm: VLLM) -> None:
    """Copy updated HF model weights into the running vllm engine in-place."""
    engine_core = llm.llm_engine.engine_core.engine_core
    engine_core.model_executor.collective_rpc(
        "reload_weights",
        kwargs={
            "weights_iterator": (
                (name, param.data.cpu())
                for name, param in hf_model.named_parameters()
            ),
            "is_checkpoint_format": True,
        },
    )


# ─── GRPO loss ────────────────────────────────────────────────────────────────

def compute_grpo_loss(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    advantages: torch.Tensor,
) -> torch.Tensor:
    full_texts = [p + r for p, r in zip(prompts, responses)]

    enc_full = tokenizer(
        full_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=tokenizer.model_max_length,
    ).to(model.device)

    enc_prompt = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=tokenizer.model_max_length,
    ).to(model.device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(**enc_full).logits          # [B, T, V]

    log_probs  = F.log_softmax(logits[:, :-1], dim=-1)
    target_ids = enc_full.input_ids[:, 1:]
    token_lp   = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    prompt_lens = enc_prompt.attention_mask.sum(dim=1)
    resp_mask   = torch.zeros_like(enc_full.attention_mask[:, 1:], dtype=torch.float)
    for i, plen in enumerate(prompt_lens):
        resp_mask[i, plen - 1:] = enc_full.attention_mask[i, plen:].float()

    seq_lp = (token_lp * resp_mask).sum(dim=1) / (resp_mask.sum(dim=1) + 1e-8)
    return -(advantages * seq_lp).mean()


# ─── Training loop ────────────────────────────────────────────────────────────

def main() -> None:
    cfg = GRPOConfig()

    # ── HF model (gradient updates) ───────────────────────────────────────
    print("Loading HF target model for training...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.target_model,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    hf_model.train()
    optimizer = torch.optim.AdamW(hf_model.parameters(), lr=cfg.learning_rate)

    # ── vllm engine (fast rollout generation with spec decode) ────────────
    print("Loading vllm inference engine with spec decode...")
    llm = VLLM(
        model=cfg.target_model,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        enforce_eager=False,
        speculative_config={
            "model": cfg.draft_model,
            "num_speculative_tokens": cfg.num_speculative_tokens,
        },
    )

    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        max_tokens=cfg.max_new_tokens,
    )

    print(f"\nReward weights: alpha={cfg.reward.alpha} (latency)  "
          f"beta={cfg.reward.beta} (tokens)  gamma={cfg.reward.gamma} (kv_mb)  "
          f"delta={cfg.reward.delta} (accept_rate)  eps={cfg.reward.eps_w} (cache_reuse)")
    print(f"Starting GRPO — {cfg.num_train_steps} steps, "
          f"batch={cfg.batch_size}, G={cfg.num_rollouts_per_prompt}\n")

    for step in range(cfg.num_train_steps):

        # ── 1. Sample prompts ─────────────────────────────────────────────
        idx = torch.randint(len(TRAIN_PROMPTS), (cfg.batch_size,)).tolist()
        batch_prompts = [TRAIN_PROMPTS[i] for i in idx]

        # ── 2. Generate G rollouts per prompt ─────────────────────────────
        all_prompts, all_responses, all_rewards = [], [], []
        all_accept, all_latency, all_kv = [], [], []

        for _ in range(cfg.num_rollouts_per_prompt):
            outputs    = llm.generate(batch_prompts, sampling_params)
            spec_stats = llm.get_spec_decode_stats()
            rewards    = compute_rewards(outputs, spec_stats, llm, cfg.reward)

            for out, r in zip(outputs, rewards):
                m = out.metrics
                stats = spec_stats.get(out.request_id, {})

                all_prompts.append(out.prompt)
                all_responses.append(out.outputs[0].text)
                all_rewards.append(r)
                all_accept.append(stats.get("accept_rate", 0.0))
                all_latency.append((m.last_token_ts - m.queued_ts) * 1e3 if m else 0.0)
                all_kv.append(_kv_memory_mb(
                    len(out.prompt_token_ids) + len(out.outputs[0].token_ids),
                    llm.llm_engine.model_config,
                ))

        # ── 3. Group-relative advantages ──────────────────────────────────
        rewards_t    = torch.tensor(all_rewards, dtype=torch.float32)
        advantages   = _znorm(rewards_t, cfg.norm_eps)

        # ── 4. GRPO gradient step ─────────────────────────────────────────
        optimizer.zero_grad()
        loss = compute_grpo_loss(
            hf_model, tokenizer,
            all_prompts, all_responses,
            advantages.to(hf_model.device),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hf_model.parameters(), cfg.grad_clip)
        optimizer.step()

        # ── 5. Sync weights to vllm ───────────────────────────────────────
        if (step + 1) % cfg.weight_sync_every == 0:
            sync_weights_to_vllm(hf_model, llm)

        # ── 6. Log ────────────────────────────────────────────────────────
        mean_r   = rewards_t.mean().item()
        accept_t = torch.tensor(all_accept)
        lat_t    = torch.tensor(all_latency)
        kv_t     = torch.tensor(all_kv)

        print(
            f"step {step+1:4d} | loss={loss.item():.4f} | "
            f"reward={mean_r:.3f} | "
            f"accept={accept_t.mean():.3f} | "
            f"latency={lat_t.mean():.0f}ms | "
            f"kv={kv_t.mean():.2f}MB"
        )

    print("\nTraining complete.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
