"""Horizon probe: does imagined reward accrue AFTER the WM has drifted?

Part A: 128 dreams (dream policy argmax, real val stacks as starts, H=40, wm_v2).
        Per-step predicted reward (raw + alive-weighted like train_dream).
Part B: Lockstep protocol (characterize_exploit.py logic, N=128): dream policy
        argmax acts on dreamed frames; identical action stream applied to real
        VecPong. Per-step WM-vs-real MSE (alive-masked) + per-step predicted reward.
Analysis: per-step lockstep MSE vs honest drift (drift_main.json mse_mean);
        fraction of imagined reward arriving after divergence > 3x honest drift;
        binned (1-10, 11-20, 21-30, 31-40); H=15/H=20 truncation counterfactual.

Run from repo root: .venv/bin/python results/diag/horizon_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from agents.policy import load_policy
from pong.env import VecPong
from wm.data import TransitionData
from wm.model import load_wm
from wm.rollout import dream

N, H = 128, 40
BINS = [(1, 10), (11, 20), (21, 30), (31, 40)]


def binned(x, agg=np.sum):
    return {f"{a}-{b}": float(agg(x[a - 1:b])) for a, b in BINS}


def main():
    config.seed_everything()
    dev = config.get_device()
    wm = load_wm(config.CKPT_DIR / "wm_v2.pt", dev, with_heads=True).eval()
    policy = load_policy(config.CKPT_DIR / "dream_agent.pt", dev).eval()

    # ---------------- Part A: dreams from real val stacks ----------------
    data = TransitionData(config.DATA_DIR / "full")
    stacks, _, _ = data.val_windows(N, H)
    stack0 = torch.from_numpy(stacks.astype(np.float32) / 255.0).to(dev)

    def act_fn(s, j):
        with torch.no_grad():
            logits, _ = policy(s)
        return logits.argmax(dim=1)

    _, rew, done_p, _ = dream(wm, stack0, H, act_fn)
    rew = rew.clamp(-1.5, 1.5).cpu().numpy()          # [N,H]
    done_p = done_p.cpu().numpy()
    conts = 1.0 - done_p
    alive = np.cumprod(
        np.concatenate([np.ones((rew.shape[0], 1), np.float32), conts[:, :-1]], 1), 1)
    r_raw = rew.mean(0)                                # per-step mean, raw
    r_aw = (rew * alive).mean(0)                       # alive-weighted (train_dream)
    r_pos = np.where(rew > 0, rew, 0).mean(0)
    r_neg = np.where(rew < 0, rew, 0).mean(0)

    # ---------------- Part B: lockstep dream vs real ----------------------
    env = VecPong(N, seed=777)
    f0 = env.reset()
    stack = torch.from_numpy(
        np.repeat(f0[:, None], 4, axis=1).astype(np.float32) / 255.0).to(dev)

    r_pred = np.zeros((N, H), np.float32)
    r_real = np.zeros((N, H), np.float32)
    hit_real = np.zeros((N, H), np.float32)
    mse_j = np.zeros(H, np.float32)
    alive_b = np.ones(N, bool)
    alive_t = np.ones((N, H), np.float32)
    dp_lock = np.zeros((N, H), np.float32)

    with torch.no_grad():
        for j in range(H):
            logits, _ = policy(stack)
            a = logits.argmax(dim=1)
            fl, r, dl = wm(stack, a)
            probs = torch.sigmoid(fl[:, 0])
            r_pred[:, j] = r.clamp(-1.5, 1.5).cpu().numpy()
            dp_lock[:, j] = torch.sigmoid(dl).cpu().numpy()
            dpix = probs.cpu().numpy()
            stack = torch.cat([stack[:, 1:], probs[:, None]], dim=1)

            f, rr, d, info = env.step(a.cpu().numpy())
            r_real[:, j] = rr
            hit_real[:, j] = info["paddle_hit"]
            alive_t[:, j] = alive_b
            diff = (f.astype(np.float32) / 255 - dpix) ** 2
            per_env = diff.mean(axis=(1, 2))
            mse_j[j] = (per_env * alive_b).sum() / max(alive_b.sum(), 1)
            alive_b &= ~d

    conts_l = 1.0 - dp_lock
    alive_l = np.cumprod(
        np.concatenate([np.ones((N, 1), np.float32), conts_l[:, :-1]], 1), 1)
    rp_raw = r_pred.mean(0)
    rp_aw = (r_pred * alive_l).mean(0)

    # ---------------- Analysis --------------------------------------------
    honest = np.array(json.loads(
        (config.RESULTS_DIR / "drift_main.json").read_text())["mse_mean"][:H])
    honest_v2 = np.array(json.loads(
        (config.RESULTS_DIR / "drift_v2.json").read_text())["mse_mean"][:H])
    ratio = mse_j / honest
    ratio_v2 = mse_j / honest_v2
    over3 = np.flatnonzero(ratio > 3.0)
    jstar = int(over3[0]) + 1 if len(over3) else None   # 1-indexed first crossing

    def frac_after(per_step, j1):
        """Fraction of total (and of positive-only) reward at steps > j1."""
        tot, pos = per_step.sum(), np.where(per_step > 0, per_step, 0).sum()
        after = per_step[j1:].sum()
        after_pos = np.where(per_step[j1:] > 0, per_step[j1:], 0).sum()
        return {"signed": float(after / tot) if abs(tot) > 1e-9 else None,
                "positive_only": float(after_pos / pos) if pos > 1e-9 else None}

    def trunc(per_step, h):
        tot, pos = per_step.sum(), np.where(per_step > 0, per_step, 0).sum()
        return {"cum_reward": float(per_step[:h].sum()),
                "frac_of_H40_signed": float(per_step[:h].sum() / tot) if abs(tot) > 1e-9 else None,
                "frac_of_H40_positive": float(
                    np.where(per_step[:h] > 0, per_step[:h], 0).sum() / pos) if pos > 1e-9 else None}

    out = {
        "n_dreams": N, "horizon": H,
        "partA_val_start_dreams": {
            "reward_per_dream_raw": float(rew.sum(1).mean()),
            "reward_per_dream_alive_weighted": float((rew * alive).sum(1).mean()),
            "per_step_reward_raw": r_raw.tolist(),
            "per_step_reward_alive_weighted": r_aw.tolist(),
            "per_step_reward_pos": r_pos.tolist(),
            "per_step_reward_neg": r_neg.tolist(),
            "binned_cum_reward_raw": binned(r_raw),
            "binned_cum_reward_alive_weighted": binned(r_aw),
            "binned_cum_reward_pos": binned(r_pos),
        },
        "partB_lockstep": {
            "imagined_reward_per_dream": float(r_pred.sum(1).mean()),
            "imagined_reward_alive_weighted": float((r_pred * alive_l).sum(1).mean()),
            "real_reward_same_actions": float((r_real * alive_t).sum(1).mean()),
            "real_hits_same_actions": float((hit_real * alive_t).sum(1).mean()),
            "per_step_mse": mse_j.tolist(),
            "per_step_reward_raw": rp_raw.tolist(),
            "per_step_reward_alive_weighted": rp_aw.tolist(),
            "binned_mean_mse": binned(mse_j, np.mean),
            "binned_cum_reward_raw": binned(rp_raw),
            "binned_cum_reward_alive_weighted": binned(rp_aw),
            "binned_cum_real_reward": binned((r_real * alive_t).mean(0)),
        },
        "honest_drift_main_first40": honest.tolist(),
        "divergence_ratio_vs_main": ratio.tolist(),
        "divergence_ratio_vs_v2": ratio_v2.tolist(),
        "binned_mean_ratio": binned(ratio, np.mean),
        "first_step_ratio_gt3": jstar,
        "frac_reward_after_3x": {
            "partA_raw": frac_after(r_raw, jstar) if jstar else None,
            "partA_alive_weighted": frac_after(r_aw, jstar) if jstar else None,
            "partB_raw": frac_after(rp_raw, jstar) if jstar else None,
            "partB_alive_weighted": frac_after(rp_aw, jstar) if jstar else None,
        },
        "truncation": {
            f"H{h}": {
                "partA_raw": trunc(r_raw, h),
                "partA_alive_weighted": trunc(r_aw, h),
                "partB_raw": trunc(rp_raw, h),
                "mse_ratio_at_h": float(ratio[h - 1]),
                "mean_ratio_first_h": float(ratio[:h].mean()),
            } for h in (10, 15, 20, 40)
        },
    }
    out_path = config.RESULTS_DIR / "diag" / "horizon_probe.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("HORIZON_PROBE_OK")
    print(json.dumps({k: out[k] for k in
                      ("first_step_ratio_gt3", "binned_mean_ratio",
                       "frac_reward_after_3x")}, indent=2))


if __name__ == "__main__":
    main()
