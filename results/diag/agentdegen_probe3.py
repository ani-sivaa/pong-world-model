"""Probe 3: WM reward-head sensitivity to policy. Dream with constant/random
action policies and compare predicted hits + return vs the learned policies.
If trivial policies also get ~0.6 predicted hits/dream, the reward signal is
policy-insensitive and provides no gradient toward tracking. Real-env truth
for the same trivial policies is computed as reference.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from pong.env import VecPong
from wm.data import TransitionData
from wm.model import load_wm
from wm.rollout import dream

DEV = config.get_device()


@torch.no_grad()
def dream_stats(wm, data, actions_fn, n=32, horizon=40):
    stacks, _, _ = data.val_windows(n, horizon)
    stack0 = torch.from_numpy(stacks.astype(np.float32) / 255.0).to(DEV)
    _, rewards, done_probs, _ = dream(wm, stack0, horizon, actions_fn)
    rw = rewards.cpu().numpy()
    alive = np.cumprod(np.concatenate(
        [np.ones((n, 1)), (1.0 - done_probs.cpu().numpy())[:, :-1]], 1), 1)
    return {
        "pred_hits_per_dream": float(((rw > 0.05) & (rw < 0.5) & (alive > 0.5)).sum(1).mean()),
        "pred_return_per_dream": float((rw * alive).sum(1).mean()),
        "mean_alive_len": float(alive.sum(1).mean()),
    }


def real_stats(actions_fn_np, n=32, steps=300, seed=123):
    env = VecPong(n, seed=seed)
    env.reset()
    hits = 0
    rng = np.random.default_rng(0)
    for t in range(steps):
        a = actions_fn_np(n, rng)
        _, _, _, info = env.step(a)
        hits += int(info["paddle_hit"].sum())
    return {"real_hits_per_40steps": hits / n / steps * 40}


def main():
    torch.manual_seed(0)
    wm = load_wm(config.CKPT_DIR / "wm_v2.pt", DEV, with_heads=True).eval()
    data = TransitionData(config.DATA_DIR / "full")
    out = {}
    pols = {
        "always_stay": (lambda s, j: torch.zeros(s.shape[0], dtype=torch.long, device=DEV),
                        lambda n, r: np.zeros(n, np.int64)),
        "always_up": (lambda s, j: torch.ones(s.shape[0], dtype=torch.long, device=DEV),
                      lambda n, r: np.ones(n, np.int64)),
        "uniform_random": (lambda s, j: torch.randint(0, 3, (s.shape[0],), device=DEV),
                           lambda n, r: r.integers(0, 3, n)),
    }
    for name, (fn_t, fn_np) in pols.items():
        d = dream_stats(wm, data, fn_t)
        d.update(real_stats(fn_np))
        out[name] = d
        print(name, json.dumps(d), flush=True)
    p = Path(__file__).parent / "agentdegen_results3.json"
    p.write_text(json.dumps(out, indent=2))
    print("WROTE", p)


if __name__ == "__main__":
    main()
