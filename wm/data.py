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


class TransitionData:
    def __init__(self, data_dir, frac: float = 1.0, seed: int = config.SEED):
        d = Path(data_dir)
        self.frames = np.load(d / "frames.npy", mmap_mode="r")     # uint8 [T,64,64]
        self.actions = np.load(d / "actions.npy")                   # uint8 [T]
        self.rewards = np.load(d / "rewards.npy").astype(np.float32)
        self.dones = np.load(d / "dones.npy")                       # bool [T]
        self.meta = json.loads((d / "meta.json").read_text())
        self.T = len(self.actions)
        self.rng = np.random.default_rng(seed)

        # episode-level split: last val_frac of episodes -> validation
        starts = np.concatenate([[0], np.flatnonzero(self.dones) + 1])
        starts = starts[starts < self.T]
        n_val_eps = max(1, int(len(starts) * config.WM["val_frac"]))
        self.val_begin = int(starts[-n_val_eps])  # stream index where val begins

        # data-frac ablation: keep only the first `frac` of TRAIN episodes
        if frac < 1.0:
            train_starts = starts[starts < self.val_begin]
            keep = max(1, int(len(train_starts) * frac))
            self.train_end = int(train_starts[keep]) if keep < len(train_starts) else self.val_begin
        else:
            self.train_end = self.val_begin

    def _valid(self, lo, hi, k):
        """Indices t in [lo+3, hi-k) whose STACK region [t-3, t-1] is done-free."""
        t = np.arange(lo + 3, hi - k)
        # done in stack region invalidates: dones[t-3] | dones[t-2] | dones[t-1]
        bad = self.dones[t - 3] | self.dones[t - 2] | self.dones[t - 1]
        return t[~bad]

    def sample(self, batch, k, val=False):
        """Returns numpy arrays: stack u8 [B,4,64,64], acts i64 [B,k],
        targets u8 [B,k,64,64], rewards f32 [B,k], dones f32 [B,k], alive f32 [B,k].
        alive[:, j] = 1 iff no done strictly before action j inside the window."""
        lo, hi = (self.val_begin, self.T) if val else (0, self.train_end)
        valid = self._valid(lo, hi, k)
        t = self.rng.choice(valid, size=batch)

        stack = self.frames[t[:, None] + np.arange(-3, 1)]          # [B,4,H,W]
        acts = self.actions[t[:, None] + np.arange(k)].astype(np.int64)
        targets = self.frames[t[:, None] + np.arange(1, k + 1)]
        rews = self.rewards[t[:, None] + np.arange(k)]
        dns = self.dones[t[:, None] + np.arange(k)].astype(np.float32)
        alive = np.cumprod(np.concatenate(
            [np.ones((batch, 1), np.float32), 1.0 - dns[:, :-1]], axis=1), axis=1)
        return stack, acts, targets, rews, dns, alive

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
