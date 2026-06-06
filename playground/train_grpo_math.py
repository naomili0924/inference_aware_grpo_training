"""
GRPO training on GSM8K with an inference-aware composite reward.

Reward function:
    reward = correctness          (1.0 if final answer matches, else 0.0)
             - alpha  * latency_ms
             - beta   * generated_tokens
             - gamma  * kv_memory_mb
             + delta  * speculative_accept_rate
             + eps_w  * cache_reuse_ratio

The model is trained to produce correct answers AND be predictable by
the draft model, making speculative decoding more efficient over time.

Adaptive curriculum via DatasetScheduler:
    - After each epoch, samples are grouped into reward buckets using KDE.
    - Harder buckets (lower mean reward) get more rollout steps:
          rollout_steps = base_rollout_steps * (1 + alpha * (1 - mean_reward))
    - A cluster report is printed after every epoch.

Usage:
    python playground/train_grpo_math.py
"""

import re
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional

import multiprocessing as mp
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import SamplingParams

from inference_aware_grpo_training import VLLM
from inference_aware_grpo_training.data_scheduler import DatasetScheduler, SchedulerConfig


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

    # dataset
    dataset: str = "openai/gsm8k"
    dataset_config: str = "main"
    max_train_samples: int = 500   # subset for quick experiments; set None for full

    # rollout — dynamic per bucket; base is for the easiest (highest-reward) bucket
    base_rollout_steps: int = 4
    rollout_alpha: float = 1.5     # harder buckets: steps = base * (1 + alpha*(1-mean_r))
    max_new_tokens: int = 512      # math needs more tokens for chain-of-thought
    temperature: float = 0.8

    # training
    learning_rate: float = 1e-6
    num_epochs: int = 20           # upper bound; early stopping may halt sooner
    batch_size: int = 4
    grad_clip: float = 1.0
    weight_sync_every: int = 1     # sync vllm weights every N steps

    # early stopping
    early_stop_patience: int = 3   # stop if eval accuracy doesn't improve for N epochs

    # adaptive curriculum scheduler
    n_buckets: int = 3
    ema_decay: float = 0.7

    # eval
    eval_every: int = 1            # eval accuracy every N epochs
    eval_samples: int = 100        # number of test problems to evaluate

    # vllm
    gpu_memory_utilization: float = 0.2
    max_model_len: int = 1024
    num_speculative_tokens: int = 5

    reward: RewardWeights = field(default_factory=RewardWeights)
    norm_eps: float = 1e-8


# ─── Math helpers ─────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "At the end of your solution, write your final answer on its own line "
    "in the format: 'The answer is <number>'\n\n"
    "Problem: {question}\n\nSolution:"
)

_ANSWER_RE = re.compile(
    r"(?:the answer is|answer:|=)\s*\$?(-?[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_GSM_RE = re.compile(r"####\s*(-?[\d,]+)")


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def extract_answer(text: str) -> Optional[str]:
    """Extract the last numerical answer from model output."""
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].replace(",", "")
    # fallback: last number in the text
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def extract_gt_answer(answer_str: str) -> Optional[str]:
    """Extract the ground-truth number from GSM8K answer field (#### N)."""
    m = _GSM_RE.search(answer_str)
    return m.group(1).replace(",", "") if m else None


def answers_match(pred: Optional[str], gt: Optional[str]) -> bool:
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        return pred.strip() == gt.strip()


# ─── Reward ───────────────────────────────────────────────────────────────────

def _kv_memory_mb(num_tokens: int, model_cfg) -> float:
    hc = model_cfg.hf_config
    bytes_per_token = (
        2 * hc.num_hidden_layers * hc.num_key_value_heads
        * (hc.hidden_size // hc.num_attention_heads) * 2  # bfloat16
    )
    return num_tokens * bytes_per_token / (1024 ** 2)


def compute_rewards(
    outputs,
    gt_answers: List[str],
    spec_stats: dict,
    llm: VLLM,
    w: RewardWeights,
) -> List[float]:
    model_cfg = llm.llm_engine.model_config
    rewards = []

    for out, gt in zip(outputs, gt_answers):
        response = out.outputs[0].text

        # task correctness
        pred = extract_answer(response)
        correctness = 1.0 if answers_match(pred, gt) else 0.0

        # latency
        m = out.metrics
        latency_ms = (m.last_token_ts - m.queued_ts) * 1e3 if m else 0.0

        # token count
        gen_tokens = len(out.outputs[0].token_ids)

        # KV memory
        kv_mb = _kv_memory_mb(len(out.prompt_token_ids) + gen_tokens, model_cfg)

        # spec accept rate
        stats = spec_stats.get(out.request_id, {})
        accept_rate = stats.get("accept_rate", 0.0)

        # prefix cache reuse
        prompt_len = len(out.prompt_token_ids)
        cached = getattr(out, "num_cached_tokens", 0) or 0
        cache_reuse = cached / prompt_len if prompt_len > 0 else 0.0

        reward = (
            correctness
            - w.alpha * latency_ms
            - w.beta  * gen_tokens
            - w.gamma * kv_mb
            + w.delta * accept_rate
            + w.eps_w * cache_reuse
        )
        rewards.append(reward)

    return rewards


def _znorm(t: torch.Tensor, eps: float) -> torch.Tensor:
    return (t - t.mean()) / (t.std() + eps)


# ─── Eval ─────────────────────────────────────────────────────────────────────

def evaluate(llm: VLLM, eval_dataset, num_samples: int, max_new_tokens: int) -> float:
    """Return accuracy on a random subset of the eval set."""
    indices = torch.randperm(len(eval_dataset))[:num_samples].tolist()
    subset = eval_dataset.select(indices)

    prompts  = [format_prompt(ex["question"]) for ex in subset]
    gt_answers = [extract_gt_answer(ex["answer"]) for ex in subset]

    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=max_new_tokens),
    )

    correct = sum(
        answers_match(extract_answer(out.outputs[0].text), gt)
        for out, gt in zip(outputs, gt_answers)
    )
    return correct / num_samples


# ─── Weight sync ──────────────────────────────────────────────────────────────

def sync_weights_to_vllm(hf_model: torch.nn.Module, llm: VLLM) -> None:
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
        logits = model(**enc_full).logits

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

    # ── Dataset ───────────────────────────────────────────────────────────
    print("Loading GSM8K...")
    ds = load_dataset(cfg.dataset, cfg.dataset_config)
    train_ds = ds["train"]
    eval_ds  = ds["test"]
    if cfg.max_train_samples:
        train_ds = train_ds.select(range(min(cfg.max_train_samples, len(train_ds))))
    print(f"  train: {len(train_ds)} examples  |  eval: {len(eval_ds)} examples")

    # ── HF model ──────────────────────────────────────────────────────────
    print("Loading HF target model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.target_model, dtype=torch.bfloat16, device_map="cuda",
    )
    hf_model.gradient_checkpointing_enable()
    hf_model.train()
    optimizer = torch.optim.AdamW(hf_model.parameters(), lr=cfg.learning_rate)

    # ── vllm engine ───────────────────────────────────────────────────────
    print("Loading vllm engine with spec decode...")
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

    # ── Adaptive curriculum scheduler ────────────────────────────────────
    sched_cfg = SchedulerConfig(
        base_rollout_steps=cfg.base_rollout_steps,
        rollout_alpha=cfg.rollout_alpha,
        ema_decay=cfg.ema_decay,
        n_buckets=cfg.n_buckets,
        batch_size=cfg.batch_size,
        min_epochs_before_bucketing=1,
    )
    scheduler = DatasetScheduler(train_ds, sched_cfg)

    # ── Baseline eval ─────────────────────────────────────────────────────
    print(f"\nBaseline eval on {cfg.eval_samples} test problems...")
    baseline_acc = evaluate(llm, eval_ds, cfg.eval_samples, cfg.max_new_tokens)
    print(f"  Baseline accuracy: {baseline_acc:.1%}\n")

    print(
        f"Starting GRPO — up to {cfg.num_epochs} epochs, batch={cfg.batch_size}, "
        f"base_rollout={cfg.base_rollout_steps}, alpha={cfg.rollout_alpha}, "
        f"patience={cfg.early_stop_patience}\n"
    )

    global_step = 0
    best_acc = baseline_acc
    epochs_without_improvement = 0

    for epoch in range(cfg.num_epochs):
        sampler = scheduler.get_sampler()
        print(f"── Epoch {epoch + 1}/{cfg.num_epochs}  "
              f"({len(sampler)} batches) ─────────────────────────")

        for batch_indices in sampler:
            global_step += 1

            # ── 1. Build batch ────────────────────────────────────────────
            batch            = train_ds.select(batch_indices)
            batch_prompts    = [format_prompt(ex["question"]) for ex in batch]
            batch_gt_answers = [extract_gt_answer(ex["answer"]) for ex in batch]

            # Dynamic rollout steps: harder bucket → more exploration
            rollout_n = sampler.rollout_steps_for_batch(batch_indices[0])

            # ── 2. Generate rollout_n rollouts per problem ────────────────
            # Collect all rollouts first so we can compute group-relative
            # advantages across the full group before any gradient step.
            rollout_prompts_list: List[List[str]] = []
            rollout_responses_list: List[List[str]] = []
            all_rewards: List[float] = []
            all_correct, all_accept, all_latency = [], [], []

            for _ in range(rollout_n):
                outputs    = llm.generate(batch_prompts, sampling_params)
                spec_stats = llm.get_spec_decode_stats()
                rewards    = compute_rewards(
                    outputs, batch_gt_answers, spec_stats, llm, cfg.reward
                )

                # Feed rewards back to scheduler for next epoch's bucketing
                scheduler.record_rewards(batch_indices, rewards)

                r_prompts, r_responses = [], []
                for i, (out, r) in enumerate(zip(outputs, rewards)):
                    m     = out.metrics
                    stats = spec_stats.get(out.request_id, {})
                    pred  = extract_answer(out.outputs[0].text)

                    r_prompts.append(out.prompt)
                    r_responses.append(out.outputs[0].text)
                    all_rewards.append(r)
                    all_correct.append(1.0 if answers_match(pred, batch_gt_answers[i]) else 0.0)
                    all_accept.append(stats.get("accept_rate", 0.0))
                    all_latency.append(
                        (m.last_token_ts - m.queued_ts) * 1e3 if m else 0.0
                    )

                rollout_prompts_list.append(r_prompts)
                rollout_responses_list.append(r_responses)

            # ── 3. Group-relative advantages ──────────────────────────────
            rewards_t  = torch.tensor(all_rewards, dtype=torch.float32)
            advantages = _znorm(rewards_t, cfg.norm_eps).to(hf_model.device)

            # ── 4. GRPO gradient step with per-rollout accumulation ───────
            # Forward/backward one rollout at a time (batch_size sequences
            # per pass) so peak memory is independent of rollout_n.
            optimizer.zero_grad()
            total_loss = 0.0
            bs = len(batch_indices)
            for k, (r_prompts, r_responses) in enumerate(
                zip(rollout_prompts_list, rollout_responses_list)
            ):
                adv_k = advantages[k * bs : (k + 1) * bs]
                loss_k = compute_grpo_loss(
                    hf_model, tokenizer, r_prompts, r_responses, adv_k
                )
                (loss_k / rollout_n).backward()   # scale so sum == mean over rollouts
                total_loss += loss_k.item()

            loss_scalar = total_loss / rollout_n
            torch.nn.utils.clip_grad_norm_(hf_model.parameters(), cfg.grad_clip)
            optimizer.step()

            # ── 5. Sync weights to vllm ───────────────────────────────────
            if global_step % cfg.weight_sync_every == 0:
                sync_weights_to_vllm(hf_model, llm)

            # ── 6. Log ────────────────────────────────────────────────────
            correct_t = torch.tensor(all_correct)
            accept_t  = torch.tensor(all_accept)
            lat_t     = torch.tensor(all_latency)

            print(
                f"  step {global_step:4d} | epoch {epoch+1} | "
                f"rollouts={rollout_n} | loss={loss_scalar:.4f} | "
                f"correct={correct_t.mean():.2f} | "
                f"accept={accept_t.mean():.3f} | "
                f"latency={lat_t.mean():.0f}ms | "
                f"reward={rewards_t.mean():.3f}"
            )

        # ── 7. End of epoch: re-bucket and report ─────────────────────────
        metrics = scheduler.end_epoch()
        print(scheduler.log_cluster_report())

        # ── 8. Periodic eval + early stopping ────────────────────────────
        if (epoch + 1) % cfg.eval_every == 0:
            acc = evaluate(llm, eval_ds, cfg.eval_samples, cfg.max_new_tokens)
            improved = acc > best_acc
            if improved:
                best_acc = acc
                epochs_without_improvement = 0
                marker = "  ★ new best"
            else:
                epochs_without_improvement += 1
                marker = f"  (no improvement {epochs_without_improvement}/{cfg.early_stop_patience})"

            print(f"\n  >>> Eval accuracy after epoch {epoch+1}: {acc:.1%}{marker}\n")

            if epochs_without_improvement >= cfg.early_stop_patience:
                print(f"Early stopping: no improvement for {cfg.early_stop_patience} epochs.")
                break

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\nTraining complete.")
    print(f"  Baseline: {baseline_acc:.1%}  →  Best: {best_acc:.1%}  (epoch {epoch+1})")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
