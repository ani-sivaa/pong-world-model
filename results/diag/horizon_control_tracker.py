"""Horizon probe CONTROL: same lockstep protocol (repeated-frame stack init,
VecPong seed 777, N=128, H=40) but actions come from the BEHAVIOR policy
(scripted tracker + 20% random) instead of the dream policy.

If tracker-action lockstep MSE is near the honest logged-action drift while the
dream-policy lockstep MSE (horizon_probe.json) is ~16x, the divergence is
policy-induced (action distribution shift), not caused by the repeated-frame
init or by dream length.

Run from repo root: .venv/bin/python results/diag/horizon_control_tracker.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from pong.env import VecPong, scripted_action
from wm.model import load_wm

N, H = 128, 40


def run(eps, dev, wm):
    env = VecPong(N, seed=777)
    f0 = env.reset()
    pol_rng = np.random.default_rng(123)
    stack = torch.from_numpy(
        np.repeat(f0[:, None], 4, axis=1).astype(np.float32) / 255.0).to(dev)
    mse_j = np.zeros(H, np.float32)
    r_pred = np.zeros((N, H), np.float32)
    alive = np.ones(N, bool)
    with torch.no_grad():
        for j in range(H):
            # tracker acts on REAL env state (in-distribution behavior policy)
            a_np = scripted_action(env.right_y, env.ball_y, eps, pol_rng)
            a = torch.from_numpy(np.asarray(a_np, np.int64)).to(dev)
            fl, r, _ = wm(stack, a)
            probs = torch.sigmoid(fl[:, 0])
            r_pred[:, j] = r.clamp(-1.5, 1.5).cpu().numpy()
            stack = torch.cat([stack[:, 1:], probs[:, None]], dim=1)
            f, rr, d, info = env.step(a_np)
            diff = (f.astype(np.float32) / 255 - probs.cpu().numpy()) ** 2
            per_env = diff.mean(axis=(1, 2))
            mse_j[j] = (per_env * alive).sum() / max(alive.sum(), 1)
            alive &= ~d
    return mse_j, r_pred


def main():
    config.seed_everything()
    dev = config.get_device()
    wm = load_wm(config.CKPT_DIR / "wm_v2.pt", dev, with_heads=True).eval()
    honest = np.array(json.loads(
        (config.RESULTS_DIR / "drift_main.json").read_text())["mse_mean"][:H])
    out = {}
    for eps in (0.20, 0.0):
        mse_j, r_pred = run(eps, dev, wm)
        out[f"eps{eps}"] = {
            "per_step_mse": mse_j.tolist(),
            "ratio_vs_main": (mse_j / honest).tolist(),
            "mean_ratio": float((mse_j / honest).mean()),
            "mse_at": {str(h): float(mse_j[h - 1]) for h in (1, 3, 5, 10, 15, 20, 40)},
            "ratio_at": {str(h): float(mse_j[h - 1] / honest[h - 1])
                         for h in (1, 3, 5, 10, 15, 20, 40)},
            "imagined_reward_per_dream": float(r_pred.sum(1).mean()),
        }
    (config.RESULTS_DIR / "diag" / "horizon_control_tracker.json").write_text(
        json.dumps(out, indent=2))
    print("CONTROL_OK")
    for k, v in out.items():
        print(k, "ratio_at:", {h: round(r, 1) for h, r in v["ratio_at"].items()},
              "mean_ratio=%.1f" % v["mean_ratio"],
              "imag_reward=%.4f" % v["imagined_reward_per_dream"])


if __name__ == "__main__":
    main()
