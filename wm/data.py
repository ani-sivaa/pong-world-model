"""Transition dataset over the contiguous-stream .npy format (see INTERFACES.md).

Windows for k-step unroll at index t use:
  stack   = frames[t-3 .. t]        (requires NO done in [t-3, t-1])
  actions = actions[t .. t+k-1]
  targets = frames[t+1 .. t+k]
  alive_j = no done in actions[t .. t+j-1]   (mask; dones INSIDE the window are
            allowed and masked, so the done head sees positive examples)
"""
import json
from pathlib import Path

import numpy as np

import config

EVENT_NAMES = {
    0: "ordinary",
    1: "hit",
    2: "score",
    3: "concede",
    4: "done_truncation",
    5: "serve_near_terminal",
    6: "boundary",
}
EVENT_CODES = {name: code for code, name in EVENT_NAMES.items()}


class TransitionData:
    def __init__(self, data_dir, frac: float = 1.0, seed: int = config.SEED,
                 bootstrap: bool = False, bootstrap_frac: float = 1.0):
        d = Path(data_dir)
        self.frames = np.load(d / "frames.npy", mmap_mode="r")     # uint8 [T,64,64]
        self.actions = np.load(d / "actions.npy")                   # uint8 [T]
        self.rewards = np.load(d / "rewards.npy").astype(np.float32)
        self.dones = np.load(d / "dones.npy")                       # bool [T]
        self.meta = json.loads((d / "meta.json").read_text())
        self.T = len(self.actions)
        self.rng = np.random.default_rng(seed)
        if bootstrap_frac <= 0:
            raise ValueError("bootstrap_frac must be positive")
        self.bootstrap = bootstrap
        self.bootstrap_frac = bootstrap_frac
        self._bootstrap_cache = {}
        events_path, priorities_path = d / "events.npy", d / "priorities.npy"
        self.has_events = events_path.exists()
        self.has_priorities = priorities_path.exists()
        self.events = (np.load(events_path, mmap_mode="r") if self.has_events
                       else np.zeros(self.T, dtype=np.uint8))
        self.priorities = (np.load(priorities_path, mmap_mode="r")
                           if self.has_priorities else np.ones(self.T, dtype=np.float32))
        if any(len(x) != self.T for x in
               (self.frames, self.rewards, self.dones, self.events, self.priorities)):
            raise ValueError(f"dataset arrays have inconsistent lengths in {d}")
        self._valid_cache = {}
        self._sampling_cache = {}

        # episode-level split: last val_frac of episodes -> validation
        starts = np.concatenate([[0], np.flatnonzero(self.dones) + 1])
        starts = starts[starts < self.T]
        n_val_eps = max(1, int(len(starts) * config.WM["val_frac"]))
        # Adaptive datasets write a natural tracker holdout last. Restrict the
        # episode-level split to it even if enrichment produced many episodes.
        natural_ranges = self.meta.get("natural_holdout_ranges", [])
        if natural_ranges:
            natural_starts = starts[starts >= int(natural_ranges[-1][0])]
            if len(natural_starts):
                n_val_eps = min(n_val_eps, len(natural_starts))
                starts_for_val = natural_starts
            else:
                starts_for_val = starts
        else:
            starts_for_val = starts
        self.val_begin = int(starts_for_val[-n_val_eps])  # episode boundary

        # data-frac ablation: keep only the first `frac` of TRAIN episodes
        if frac < 1.0:
            train_starts = starts[starts < self.val_begin]
            keep = max(1, int(len(train_starts) * frac))
            self.train_end = int(train_starts[keep]) if keep < len(train_starts) else self.val_begin
        else:
            self.train_end = self.val_begin

    def _valid(self, lo, hi, k):
        """Indices t in [lo+3, hi-k) whose STACK region [t-3, t-1] is done-free."""
        key = (int(lo), int(hi), int(k))
        if key in self._valid_cache:
            return self._valid_cache[key]
        t = np.arange(lo + 3, hi - k)
        # done in stack region invalidates: dones[t-3] | dones[t-2] | dones[t-1]
        bad = self.dones[t - 3] | self.dones[t - 2] | self.dones[t - 1]
        self._valid_cache[key] = t[~bad]
        return self._valid_cache[key]

    def _window_event(self, valid, k):
        """Primary event for each window, selected by configured event importance."""
        weights = config.FLYWHEEL["event_priority"]
        rank = np.asarray([weights.get(EVENT_NAMES[i], 1.0)
                           for i in range(max(EVENT_NAMES) + 1)], np.float32)
        ev = np.asarray(self.events[valid[:, None] + np.arange(k)])
        safe = np.minimum(ev, len(rank) - 1)
        best = np.argmax(rank[safe], axis=1)
        return ev[np.arange(len(valid)), best].astype(np.uint8)

    def _sample_indices(self, valid, batch, k, sampler):
        key = (int(valid[0]), int(valid[-1]), len(valid), int(k), sampler)
        probabilities = self._sampling_cache.get(key)
        if probabilities is None:
            if sampler == "natural" or (sampler == "balanced" and not self.has_events) \
                    or (sampler == "priority" and not self.has_priorities):
                p = np.ones(len(valid), dtype=np.float64)
            elif sampler == "priority":
                p = np.max(np.asarray(
                    self.priorities[valid[:, None] + np.arange(k)],
                    dtype=np.float64), axis=1)
            elif sampler == "balanced":
                primary = self._window_event(valid, k)
                counts = np.bincount(primary, minlength=max(EVENT_NAMES) + 1)
                targets = config.FLYWHEEL["balanced_event_weight"]
                target = np.asarray([targets.get(EVENT_NAMES[i], 0.0)
                                     for i in range(len(counts))], np.float64)
                per_class = np.divide(target, counts, out=np.zeros_like(target),
                                      where=counts > 0)
                p = per_class[np.minimum(primary, len(per_class) - 1)]
            else:
                raise ValueError(
                    f"unknown sampler {sampler!r}; expected natural, balanced, or priority")
            p = np.maximum(p, 0.0)
            if self.bootstrap:
                # A fixed, seed-specific nonparametric bootstrap over complete
                # valid windows. Multiplicity weights preserve contiguous
                # stacks/unroll targets while diversifying ensemble members.
                bootstrap_key = (int(valid[0]), int(valid[-1]), len(valid), int(k))
                multiplicity = self._bootstrap_cache.get(bootstrap_key)
                if multiplicity is None:
                    draws = max(1, int(round(len(valid) * self.bootstrap_frac)))
                    positions = self.rng.integers(0, len(valid), size=draws)
                    multiplicity = np.bincount(
                        positions, minlength=len(valid)).astype(np.float64)
                    self._bootstrap_cache[bootstrap_key] = multiplicity
                p *= multiplicity
            probabilities = p / p.sum() if p.sum() > 0 else None
            self._sampling_cache[key] = probabilities
        return self.rng.choice(valid, size=batch, p=probabilities)

    def sample(self, batch, k, val=False, sampler="natural", return_info=False):
        """Returns numpy arrays: stack u8 [B,4,64,64], acts i64 [B,k],
        targets u8 [B,k,64,64], rewards f32 [B,k], dones f32 [B,k], alive f32 [B,k].
        alive[:, j] = 1 iff no done strictly before action j inside the window.

        Training may use ``sampler="balanced"`` (inverse event frequency) or
        ``sampler="priority"`` (saved per-transition priorities). Validation is
        always natural. Existing datasets and callers retain the original tuple.
        With return_info=True, returns ``(batch_tuple, info)`` where info contains
        contiguous window indices and event labels."""
        lo, hi = (self.val_begin, self.T) if val else (0, self.train_end)
        valid = self._valid(lo, hi, k)
        if len(valid) == 0:
            raise ValueError(f"no valid {k}-step windows in {'validation' if val else 'train'} split")
        t = self._sample_indices(valid, batch, k, "natural" if val else sampler)

        stack = self.frames[t[:, None] + np.arange(-3, 1)]          # [B,4,H,W]
        acts = self.actions[t[:, None] + np.arange(k)].astype(np.int64)
        targets = self.frames[t[:, None] + np.arange(1, k + 1)]
        rews = self.rewards[t[:, None] + np.arange(k)]
        dns = self.dones[t[:, None] + np.arange(k)].astype(np.float32)
        alive = np.cumprod(np.concatenate(
            [np.ones((batch, 1), np.float32), 1.0 - dns[:, :-1]], axis=1), axis=1)
        result = (stack, acts, targets, rews, dns, alive)
        if not return_info:
            return result
        event_windows = np.asarray(self.events[t[:, None] + np.arange(k)], dtype=np.uint8)
        return result, {
            "indices": t,
            "events": event_windows,
            "primary_events": self._window_event(t, k),
            "priorities": np.asarray(self.priorities[t], dtype=np.float32),
        }

    def val_windows(self, n, horizon):
        """n fixed (stack, actions, real_frames) windows of `horizon` steps from
        val episodes with NO done inside — for drift curves / rollout GIFs."""
        t = np.arange(self.val_begin + 3, self.T - horizon)
        if len(t) == 0:
            raise ValueError(f"no val windows of horizon {horizon}")
        # reject any window containing a done anywhere in [t-3, t+horizon-1]
        done_cum = np.cumsum(np.concatenate([[0], self.dones.astype(np.int64)]))
        n_dones = done_cum[t + horizon] - done_cum[t - 3]
        t = t[n_dones == 0]
        if len(t) == 0:
            raise ValueError(f"no done-free val windows of horizon {horizon}")
        rng = np.random.default_rng(0)  # fixed: same windows for every model/ablation
        t = rng.choice(t, size=min(n, len(t)), replace=False)
        stacks = self.frames[t[:, None] + np.arange(-3, 1)]
        acts = self.actions[t[:, None] + np.arange(horizon)].astype(np.int64)
        real = self.frames[t[:, None] + np.arange(1, horizon + 1)]
        return stacks, acts, real
