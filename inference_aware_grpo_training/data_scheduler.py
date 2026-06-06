"""
DatasetScheduler — adaptive reward-based curriculum for GRPO training.

After each epoch the scheduler:
  1. Collects per-sample EMA rewards from the epoch buffer.
  2. Runs KDE on the reward distribution to find natural valley boundaries
     (adaptive buckets that shift as the model improves).
  3. Groups samples by bucket and assigns rollout steps inversely proportional
     to the bucket's mean reward — harder samples get more exploration.
  4. Returns a ScheduledBatchSampler that the DataLoader can use directly.

Usage
-----
    scheduler = DatasetScheduler(dataset, cfg)

    for epoch in range(cfg.num_epochs):
        sampler = scheduler.get_sampler()          # fresh sampler each epoch
        loader  = DataLoader(dataset, batch_sampler=sampler)

        for batch in loader:
            outputs = rollout(batch, rollout_steps=sampler.rollout_steps_for(batch))
            rewards = compute_rewards(outputs, batch["gt_answers"], ...)
            scheduler.record_rewards(batch["indices"], rewards)   # <-- feed back

        scheduler.end_epoch()                      # triggers re-bucketing
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset, Sampler


# ---------------------------------------------------------------------------
# Cluster Metrics
# ---------------------------------------------------------------------------

@dataclass
class BucketMetrics:
    """Stats for a single bucket after one epoch."""
    bucket_id:    int
    n_samples:    int
    mean_reward:  float
    std_reward:   float
    min_reward:   float
    max_reward:   float
    median_reward: float
    p25_reward:   float          # 25th percentile
    p75_reward:   float          # 75th percentile
    boundary_low: float          # lower reward boundary (inclusive)
    boundary_high: float         # upper reward boundary (exclusive / 1.0 for last)
    rollout_steps: int
    pct_of_total: float          # fraction of dataset in this bucket


@dataclass
class EpochClusterMetrics:
    """Full clustering snapshot after one epoch."""
    epoch:           int
    n_buckets_actual: int         # may be < cfg.n_buckets if not enough valleys
    boundaries:      List[float]  # split points between buckets
    buckets:         List[BucketMetrics]

    # Distribution-level stats (over all samples)
    global_mean:     float
    global_std:      float
    global_min:      float
    global_max:      float

    # Spread: how separated are the bucket means?
    # High = clusters are well-separated; low = distribution still collapsed
    inter_bucket_spread: float    # std of per-bucket means
    # Balance: are buckets roughly equal in size?
    # 1.0 = perfectly balanced; near 0 = one bucket dominates
    bucket_balance:  float        # 1 - (std(bucket_sizes) / mean(bucket_sizes))

    def wandb_dict(self) -> Dict[str, float]:
        """Flat dict ready for wandb.log() or any key-value logger."""
        d: Dict[str, float] = {
            "scheduler/n_buckets":           self.n_buckets_actual,
            "scheduler/global_mean_reward":  self.global_mean,
            "scheduler/global_std_reward":   self.global_std,
            "scheduler/inter_bucket_spread": self.inter_bucket_spread,
            "scheduler/bucket_balance":      self.bucket_balance,
        }
        for b in self.buckets:
            prefix = f"scheduler/bucket_{b.bucket_id}"
            d[f"{prefix}/n_samples"]    = b.n_samples
            d[f"{prefix}/pct_of_total"] = b.pct_of_total
            d[f"{prefix}/mean_reward"]  = b.mean_reward
            d[f"{prefix}/std_reward"]   = b.std_reward
            d[f"{prefix}/median"]       = b.median_reward
            d[f"{prefix}/p25"]          = b.p25_reward
            d[f"{prefix}/p75"]          = b.p75_reward
            d[f"{prefix}/rollout_steps"]= b.rollout_steps
        return d


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    # Rollout steps
    base_rollout_steps: int = 8
    """Rollout steps assigned to the easiest (highest-reward) bucket."""

    rollout_alpha: float = 1.5
    """
    Scaling factor for harder buckets.
    rollout_steps = base * (1 + alpha * (1 - mean_reward))
    At alpha=1.5, a bucket with mean_reward=0 gets base * 2.5 steps;
    a bucket with mean_reward=1 gets exactly base steps.
    """

    # Reward tracking
    ema_decay: float = 0.7
    """
    EMA decay for per-sample reward history.
    0.0 = only use latest reward; 1.0 = never update from training.
    """

    min_epochs_before_bucketing: int = 1
    """
    Epochs to collect rewards before adaptive bucketing activates.
    During warm-up, all samples are in one bucket.
    """

    # Bucketing
    n_buckets: int = 3
    """Target number of adaptive buckets."""

    kde_bandwidth: float = 0.05
    """
    KDE bandwidth for reward density estimation (range 0–1).
    Smaller = more sensitive to local structure.
    """

    min_bucket_size: int = 8
    """
    Minimum samples per bucket. Tiny buckets are merged into their neighbour.
    """

    # Batching
    batch_size: int = 16
    intra_bucket_shuffle: bool = True


# ---------------------------------------------------------------------------
# Reward Tracker
# ---------------------------------------------------------------------------

class RewardTracker:
    """
    Maintains a per-sample EMA reward and collects an epoch-level buffer.
    """

    def __init__(self, n_samples: int, ema_decay: float = 0.7):
        self.n_samples = n_samples
        self.ema_decay = ema_decay
        self._ema: np.ndarray = np.full(n_samples, 0.5)   # neutral init
        self._seen: np.ndarray = np.zeros(n_samples, dtype=bool)
        self._epoch_buf: dict[int, list[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Called during training
    # ------------------------------------------------------------------

    def record(self, indices: Sequence[int], rewards: Sequence[float]) -> None:
        """Record a batch of (index, reward) pairs during an epoch."""
        for idx, r in zip(indices, rewards):
            self._epoch_buf[int(idx)].append(float(r))

    # ------------------------------------------------------------------
    # Called at end of epoch
    # ------------------------------------------------------------------

    def flush_epoch(self) -> np.ndarray:
        """
        Aggregate epoch buffer into EMA, clear buffer, return current EMA array.
        """
        for idx, rewards in self._epoch_buf.items():
            epoch_mean = float(np.mean(rewards))
            if self._seen[idx]:
                self._ema[idx] = (
                    self.ema_decay * self._ema[idx]
                    + (1.0 - self.ema_decay) * epoch_mean
                )
            else:
                self._ema[idx] = epoch_mean
                self._seen[idx] = True
        self._epoch_buf.clear()
        return self._ema.copy()

    @property
    def ema_rewards(self) -> np.ndarray:
        return self._ema.copy()

    @property
    def n_seen(self) -> int:
        return int(self._seen.sum())


# ---------------------------------------------------------------------------
# Adaptive Bucketizer
# ---------------------------------------------------------------------------

class AdaptiveBucketizer:
    """
    Uses Gaussian KDE on the reward distribution to find natural valley
    boundaries, then assigns each sample to a bucket.

    Buckets are re-computed every time `fit()` is called (i.e. every epoch).
    """

    def __init__(
        self,
        n_buckets: int = 3,
        bandwidth: float = 0.05,
        min_bucket_size: int = 8,
    ):
        self.n_buckets = n_buckets
        self.bandwidth = bandwidth
        self.min_bucket_size = min_bucket_size
        self.boundaries_: Optional[List[float]] = None  # learned after fit()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _kde_density(self, rewards: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate Gaussian KDE at points x."""
        bw = self.bandwidth
        diffs = (x[:, None] - rewards[None, :]) / bw       # (M, N)
        kernel = np.exp(-0.5 * diffs ** 2) / (bw * math.sqrt(2 * math.pi))
        return kernel.mean(axis=1)

    def _find_valleys(self, rewards: np.ndarray) -> List[float]:
        """
        Evaluate KDE on a fine grid, find the (n_buckets-1) deepest valleys
        in [0, 1] as bucket split points.
        """
        grid = np.linspace(0.0, 1.0, 512)
        density = self._kde_density(rewards, grid)

        # Find local minima (valleys)
        valleys = []
        for i in range(1, len(grid) - 1):
            if density[i] < density[i - 1] and density[i] < density[i + 1]:
                valleys.append((density[i], grid[i]))

        if not valleys:
            # Fallback: uniform split
            return [i / self.n_buckets for i in range(1, self.n_buckets)]

        # Sort by density depth (deepest valleys first) and take top k
        valleys.sort(key=lambda v: v[0])
        boundaries = sorted(v[1] for v in valleys[: self.n_buckets - 1])
        return boundaries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, rewards: np.ndarray) -> "AdaptiveBucketizer":
        """Compute adaptive bucket boundaries from current reward distribution."""
        rewards_clipped = np.clip(rewards, 0.0, 1.0)
        self.boundaries_ = self._find_valleys(rewards_clipped)
        return self

    def assign(self, rewards: np.ndarray) -> np.ndarray:
        """
        Assign each sample to a bucket index (0 = hardest, K-1 = easiest).
        """
        if self.boundaries_ is None:
            raise RuntimeError("Call fit() before assign().")
        boundaries = self.boundaries_
        bucket_ids = np.zeros(len(rewards), dtype=int)
        for b in boundaries:
            bucket_ids += (rewards > b).astype(int)
        return bucket_ids

    def bucket_summary(
        self, rewards: np.ndarray, bucket_ids: np.ndarray
    ) -> List[dict]:
        """Return per-bucket stats for logging."""
        boundaries = self.boundaries_ or []
        # Build boundary ranges per bucket — always produce n_buckets+1 edges
        edges = [0.0] + list(boundaries) + [1.0]
        # Pad edges to ensure index b and b+1 always exist for any bucket id
        while len(edges) < self.n_buckets + 1:
            edges.append(1.0)
        n_total = len(rewards)
        summary = []
        for b in range(self.n_buckets):
            mask = bucket_ids == b
            r = rewards[mask]
            if len(r) == 0:
                summary.append({
                    "bucket": b, "n": 0,
                    "mean_reward": 0.0, "std_reward": 0.0,
                    "min_reward": 0.0,  "max_reward": 0.0,
                    "median_reward": 0.0,
                    "p25_reward": 0.0,  "p75_reward": 0.0,
                    "boundary_low": edges[b], "boundary_high": edges[b + 1],
                    "pct_of_total": 0.0,
                })
                continue
            summary.append({
                "bucket":        b,
                "n":             int(mask.sum()),
                "mean_reward":   float(r.mean()),
                "std_reward":    float(r.std()),
                "min_reward":    float(r.min()),
                "max_reward":    float(r.max()),
                "median_reward": float(np.median(r)),
                "p25_reward":    float(np.percentile(r, 25)),
                "p75_reward":    float(np.percentile(r, 75)),
                "boundary_low":  edges[b],
                "boundary_high": edges[b + 1],
                "pct_of_total":  float(mask.sum()) / n_total,
            })
        return summary


# ---------------------------------------------------------------------------
# Rollout Step Calculator
# ---------------------------------------------------------------------------

def rollout_steps_for_bucket(
    mean_reward: float,
    base_rollout_steps: int,
    alpha: float,
) -> int:
    """
    Inversely scale rollout steps by mean reward.

        steps = base * (1 + alpha * (1 - mean_reward))

    mean_reward=0.0  →  base * (1 + alpha)   [most exploration]
    mean_reward=1.0  →  base * 1              [least exploration]
    """
    multiplier = 1.0 + alpha * (1.0 - mean_reward)
    return max(1, round(base_rollout_steps * multiplier))


# ---------------------------------------------------------------------------
# ScheduledBatchSampler
# ---------------------------------------------------------------------------

class ScheduledBatchSampler(Sampler):
    """
    A batch sampler that:
      - Groups samples by bucket (similar reward → same batch).
      - Shuffles within each bucket independently.
      - Carries the per-bucket rollout step count so the training loop
        can query it.

    Usage in training loop
    ----------------------
        for batch_indices in sampler:
            rollout_n = sampler.rollout_steps_for_batch(batch_indices[0])
            batch = dataset[batch_indices]
            outputs = rollout(batch, rollout_steps=rollout_n)
    """

    def __init__(
        self,
        bucket_to_indices: dict[int, List[int]],
        bucket_to_rollout: dict[int, int],
        batch_size: int,
        shuffle: bool = True,
    ):
        self.bucket_to_indices = bucket_to_indices
        self.bucket_to_rollout = bucket_to_rollout
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Build a flat index → rollout map for O(1) lookup
        self._idx_rollout: dict[int, int] = {}
        for b, idxs in bucket_to_indices.items():
            for i in idxs:
                self._idx_rollout[i] = bucket_to_rollout[b]

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[List[int]]:
        all_batches: List[List[int]] = []
        for b, idxs in self.bucket_to_indices.items():
            pool = list(idxs)
            if self.shuffle:
                random.shuffle(pool)
            for start in range(0, len(pool), self.batch_size):
                batch = pool[start : start + self.batch_size]
                if batch:
                    all_batches.append(batch)
        if self.shuffle:
            random.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        return sum(
            math.ceil(len(idxs) / self.batch_size)
            for idxs in self.bucket_to_indices.values()
        )

    # ------------------------------------------------------------------
    # Rollout query helpers
    # ------------------------------------------------------------------

    def rollout_steps_for_index(self, sample_idx: int) -> int:
        return self._idx_rollout.get(sample_idx, 1)

    def rollout_steps_for_batch(self, any_index_in_batch: int) -> int:
        """All samples in a bucket share the same rollout count."""
        return self.rollout_steps_for_index(any_index_in_batch)


# ---------------------------------------------------------------------------
# DatasetScheduler  (main entry point)
# ---------------------------------------------------------------------------

class DatasetScheduler:
    """
    Wraps a HuggingFace / PyTorch dataset and produces a fresh
    ScheduledBatchSampler after each epoch.

    Parameters
    ----------
    dataset : Dataset
        Must have a ``__len__``. Samples are identified by integer index.
    cfg     : SchedulerConfig
    """

    def __init__(self, dataset: Dataset, cfg: SchedulerConfig):
        self.dataset = dataset
        self.cfg = cfg
        self._n = len(dataset)  # type: ignore[arg-type]
        self._epoch = 0

        self.tracker = RewardTracker(self._n, ema_decay=cfg.ema_decay)
        self.bucketizer = AdaptiveBucketizer(
            n_buckets=cfg.n_buckets,
            bandwidth=cfg.kde_bandwidth,
            min_bucket_size=cfg.min_bucket_size,
        )

        # State set after end_epoch()
        self._last_bucket_ids: Optional[np.ndarray] = None
        self._last_summary: Optional[List[dict]] = None
        self._last_ema: Optional[np.ndarray] = None
        self._last_metrics: Optional[EpochClusterMetrics] = None

    # ------------------------------------------------------------------
    # Called during training
    # ------------------------------------------------------------------

    def record_rewards(
        self, indices: Sequence[int], rewards: Sequence[float]
    ) -> None:
        """
        Feed back rewards for a batch. Call this after compute_rewards()
        every training step.
        """
        self.tracker.record(indices, rewards)

    # ------------------------------------------------------------------
    # Called at end of each epoch
    # ------------------------------------------------------------------

    def end_epoch(self) -> EpochClusterMetrics:
        """
        Flush the epoch reward buffer, re-fit bucket boundaries, compute
        cluster metrics, and return an EpochClusterMetrics snapshot.
        """
        ema = self.tracker.flush_epoch()
        self._epoch += 1
        self._last_ema = ema

        if self._epoch < self.cfg.min_epochs_before_bucketing:
            bucket_ids = np.zeros(self._n, dtype=int)
            boundaries: List[float] = []
        else:
            self.bucketizer.fit(ema)
            bucket_ids = self.bucketizer.assign(ema)
            boundaries = list(self.bucketizer.boundaries_ or [])

        self._last_bucket_ids = bucket_ids
        summary = self.bucketizer.bucket_summary(ema, bucket_ids) \
            if self._epoch >= self.cfg.min_epochs_before_bucketing \
            else [{
                "bucket": 0, "n": self._n,
                "mean_reward":   float(ema.mean()),
                "std_reward":    float(ema.std()),
                "min_reward":    float(ema.min()),
                "max_reward":    float(ema.max()),
                "median_reward": float(np.median(ema)),
                "p25_reward":    float(np.percentile(ema, 25)),
                "p75_reward":    float(np.percentile(ema, 75)),
                "boundary_low":  0.0, "boundary_high": 1.0,
                "pct_of_total":  1.0,
            }]
        self._last_summary = summary

        # Build BucketMetrics list
        cfg = self.cfg
        bucket_metrics = []
        for s in summary:
            b = s["bucket"]
            rollout_n = rollout_steps_for_bucket(
                s["mean_reward"], cfg.base_rollout_steps, cfg.rollout_alpha
            )
            bucket_metrics.append(BucketMetrics(
                bucket_id=b,
                n_samples=s["n"],
                mean_reward=s["mean_reward"],
                std_reward=s["std_reward"],
                min_reward=s["min_reward"],
                max_reward=s["max_reward"],
                median_reward=s["median_reward"],
                p25_reward=s["p25_reward"],
                p75_reward=s["p75_reward"],
                boundary_low=s["boundary_low"],
                boundary_high=s["boundary_high"],
                rollout_steps=rollout_n,
                pct_of_total=s["pct_of_total"],
            ))

        # Distribution-level stats
        means = [b.mean_reward for b in bucket_metrics if b.n_samples > 0]
        sizes = [b.n_samples   for b in bucket_metrics if b.n_samples > 0]
        inter_spread = float(np.std(means)) if len(means) > 1 else 0.0
        if len(sizes) > 1 and np.mean(sizes) > 0:
            balance = float(1.0 - np.std(sizes) / np.mean(sizes))
        else:
            balance = 1.0

        metrics = EpochClusterMetrics(
            epoch=self._epoch,
            n_buckets_actual=len([b for b in bucket_metrics if b.n_samples > 0]),
            boundaries=boundaries,
            buckets=bucket_metrics,
            global_mean=float(ema.mean()),
            global_std=float(ema.std()),
            global_min=float(ema.min()),
            global_max=float(ema.max()),
            inter_bucket_spread=inter_spread,
            bucket_balance=balance,
        )
        self._last_metrics = metrics
        return metrics

    # ------------------------------------------------------------------
    # Called at start of each epoch
    # ------------------------------------------------------------------

    def get_sampler(self) -> ScheduledBatchSampler:
        """
        Build and return a ScheduledBatchSampler for the next epoch.
        Safe to call before the first end_epoch() — returns a uniform sampler.
        """
        cfg = self.cfg

        if self._last_bucket_ids is None:
            # Before any bucketing: all samples in bucket 0
            bucket_ids = np.zeros(self._n, dtype=int)
            mean_rewards = {0: 0.5}
        else:
            bucket_ids = self._last_bucket_ids
            ema = self.tracker.ema_rewards
            mean_rewards = {}
            for b in range(cfg.n_buckets):
                mask = bucket_ids == b
                r = ema[mask]
                mean_rewards[b] = float(r.mean()) if len(r) else 0.5

        # Build bucket → [indices] map
        bucket_to_indices: dict[int, List[int]] = defaultdict(list)
        for idx, b in enumerate(bucket_ids):
            bucket_to_indices[int(b)].append(idx)

        # Compute rollout steps per bucket
        bucket_to_rollout: dict[int, int] = {}
        for b, mr in mean_rewards.items():
            bucket_to_rollout[b] = rollout_steps_for_bucket(
                mean_reward=mr,
                base_rollout_steps=cfg.base_rollout_steps,
                alpha=cfg.rollout_alpha,
            )

        return ScheduledBatchSampler(
            bucket_to_indices=dict(bucket_to_indices),
            bucket_to_rollout=bucket_to_rollout,
            batch_size=cfg.batch_size,
            shuffle=cfg.intra_bucket_shuffle,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def cluster_metrics(self) -> Optional[EpochClusterMetrics]:
        """Full EpochClusterMetrics from the last end_epoch() call."""
        return self._last_metrics

    @property
    def bucket_summary(self) -> Optional[List[dict]]:
        """Raw per-bucket dicts from the last end_epoch() call."""
        return self._last_summary

    def log_state(self) -> str:
        """Compact one-liner per bucket for quick console logging."""
        if not self._last_metrics:
            return f"DatasetScheduler | epoch={self._epoch} | warm-up"
        m = self._last_metrics
        lines = [
            f"DatasetScheduler | epoch={m.epoch} | "
            f"global mean={m.global_mean:.3f} ± {m.global_std:.3f} | "
            f"spread={m.inter_bucket_spread:.3f} | balance={m.bucket_balance:.2f}"
        ]
        for b in m.buckets:
            lines.append(
                f"  bucket {b.bucket_id} [{b.boundary_low:.2f}–{b.boundary_high:.2f}]: "
                f"n={b.n_samples:5d} ({b.pct_of_total*100:4.1f}%) | "
                f"mean={b.mean_reward:.3f} ± {b.std_reward:.3f} | "
                f"[{b.min_reward:.3f}, {b.median_reward:.3f}, {b.max_reward:.3f}] | "
                f"rollout={b.rollout_steps}"
            )
        return "\n".join(lines)

    def log_cluster_report(self, bar_width: int = 40) -> str:
        """
        Print a detailed ASCII cluster report with:
          - Per-bucket reward histogram (ASCII bar chart)
          - IQR box per bucket  [p25 ---|median|--- p75]
          - Boundary markers on a shared reward axis
          - Summary health indicators (spread + balance)

        Example output
        --------------
        ╔══ Epoch 3 — Cluster Report ══════════════════════════╗
        │ global  mean=0.421 std=0.198  min=0.031  max=0.947   │
        │ spread (inter-bucket)=0.241   balance=0.87            │
        ╠════════════════════════════════════════════════════════╣
        │ bucket 0  [0.00–0.31]  n=1823 (36.5%)  rollout=18    │
        │ mean=0.182 ± 0.071                                     │
        │ IQR  [0.13 ──|0.18|── 0.24]                           │
        │ ████████████████░░░░░░░░░░░░░░░░░░░░░░░░  (density)   │
        ...
        """
        if not self._last_metrics or self._last_ema is None:
            return f"DatasetScheduler | epoch={self._epoch} | no metrics yet"

        m = self._last_metrics
        ema = self._last_ema
        W = bar_width
        SEP = "─" * (W + 44)

        lines: List[str] = []
        lines.append(f"╔══ Epoch {m.epoch} — Cluster Report {'═' * (W + 26)}╗")
        lines.append(
            f"│ global  mean={m.global_mean:.3f}  std={m.global_std:.3f}"
            f"  min={m.global_min:.3f}  max={m.global_max:.3f}"
        )
        health_spread = (
            "▲ clusters separating" if m.inter_bucket_spread > 0.15
            else "▼ distribution still collapsed"
        )
        health_balance = (
            "✓ balanced" if m.bucket_balance > 0.6
            else "! imbalanced — consider reducing n_buckets"
        )
        lines.append(
            f"│ spread={m.inter_bucket_spread:.3f} {health_spread}"
            f"   balance={m.bucket_balance:.2f} {health_balance}"
        )

        if m.boundaries:
            b_str = "  |  ".join(f"{v:.3f}" for v in m.boundaries)
            lines.append(f"│ boundaries: {b_str}")

        lines.append("╠" + SEP)

        for bm in m.buckets:
            lines.append(
                f"│ bucket {bm.bucket_id}  [{bm.boundary_low:.2f}–{bm.boundary_high:.2f}]"
                f"  n={bm.n_samples:5d} ({bm.pct_of_total*100:4.1f}%)"
                f"  rollout={bm.rollout_steps}"
            )
            lines.append(
                f"│   mean={bm.mean_reward:.3f} ± {bm.std_reward:.3f}"
                f"   median={bm.median_reward:.3f}"
            )

            # IQR display: [p25 ─────|median|───── p75]
            rng = bm.boundary_high - bm.boundary_low
            if rng > 0:
                def _pos(v: float) -> int:
                    return round((v - bm.boundary_low) / rng * 20)
                p25_pos = _pos(bm.p25_reward)
                med_pos = _pos(bm.median_reward)
                p75_pos = _pos(bm.p75_reward)
                iqr_line = [" "] * 22
                for i in range(max(0, p25_pos), min(21, p75_pos + 1)):
                    iqr_line[i] = "─"
                if 0 <= med_pos <= 21:
                    iqr_line[med_pos] = "┼"
                iqr_str = "".join(iqr_line)
                lines.append(
                    f"│   IQR [{bm.boundary_low:.2f}  {iqr_str}  {bm.boundary_high:.2f}]"
                    f"  p25={bm.p25_reward:.3f}  p75={bm.p75_reward:.3f}"
                )

            # ASCII histogram within this bucket's reward range
            mask = (self._last_bucket_ids == bm.bucket_id)
            bucket_rewards = ema[mask]
            if len(bucket_rewards) > 0 and rng > 0:
                bins = np.linspace(bm.boundary_low, bm.boundary_high, W + 1)
                counts, _ = np.histogram(bucket_rewards, bins=bins)
                max_count = counts.max() if counts.max() > 0 else 1
                bar = ""
                for c in counts:
                    frac = c / max_count
                    if frac > 0.66:
                        bar += "█"
                    elif frac > 0.33:
                        bar += "▓"
                    elif frac > 0.0:
                        bar += "░"
                    else:
                        bar += " "
                lines.append(f"│   hist [{bar}]")

            lines.append("│" + SEP)

        lines.append("╚" + "═" * (W + 44) + "╝")
        return "\n".join(lines)