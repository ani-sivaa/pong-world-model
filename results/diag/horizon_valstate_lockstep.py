"""Horizon probe Part C — lockstep from PROPER mid-episode val states.

Reconstructs VecPong state (ball pos/vel, paddle ys) from the 128 val-window
frame stacks used in Part A, then runs three lockstep streams for H=40:

  1. logged actions   -> reconstructed env vs LOGGED real frames
                         (reconstruction error floor + honest-drift replication:
                          WM with logged actions vs logged frames)
  2. dream policy argmax (acting on DREAMED frames) -> WM vs reconstructed env
                         (policy-conditional divergence from real states)
  3. tracker actions (eps=0.2, acting on env state) -> WM vs reconstructed env
                         (protocol-matched in-distribution reference)

Run from repo root: .venv/bin/python results/diag/horizon_valstate_lockstep.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from agents.policy import load_policy
from pong.env import VecPong, scripted_action
from wm.data import TransitionData
from wm.model import load_wm

N, H = 128, 40
E = config.ENV
BINS = [(1, 10), (11, 20), (21, 30), (31, 40)]


def _runs(col):
    """White runs (start,len) in a 0/255 column vector rows 1..62."""
    w = np.flatnonzero(col[1:63] == 255) + 1
    if len(w) == 0:
        return []
    breaks = np.flatnonzero(np.diff(w) > 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(w) - 1]])
    return [(int(w[s]), int(w[e] - w[s] + 1)) for s, e in zip(starts, ends)]


def _paddle_y(frame, col):
    runs = _runs(frame[:, col])
    if not runs:
        return None
    # paddle = the run closest to paddle_h tall (ball adds a <=2 run or merges)
    best = min(runs, key=lambda r: abs(r[1] - E["paddle_h"]))
    return float(best[0]) if best[1] >= E["paddle_h"] else float(best[0])


def _ball_xy(frame):
    f = frame.copy()
    f[0, :] = 0
    f[-1, :] = 0
    f[:, E["left_x"]:E["left_x"] + E["paddle_w"]] = 0
    f[:, E["right_x"]:E["right_x"] + E["paddle_w"]] = 0
    ys, xs = np.nonzero(f == 255)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min())


def reconstruct(stacks):
    """stacks u8 [N,4,64,64] -> state dict + keep mask."""
    n = stacks.shape[0]
    st = dict(ball_x=np.zeros(n), ball_y=np.zeros(n), ball_vx=np.zeros(n),
              ball_vy=np.zeros(n), left_y=np.zeros(n), right_y=np.zeros(n))
    keep = np.ones(n, bool)
    for i in range(n):
        f3, f2 = stacks[i, 3], stacks[i, 2]           # newest, previous
        ly, ry = _paddle_y(f3, E["left_x"]), _paddle_y(f3, E["right_x"])
        b3, b2 = _ball_xy(f3), _ball_xy(f2)
        if ly is None or ry is None or b3 is None or b2 is None:
            keep[i] = False
            continue
        x3, y3 = b3
        x2, y2 = b2
        vx = np.sign(x3 - x2) * E["ball_vx"] if x3 != x2 else E["ball_vx"]
        # vy: prefer 2-frame diff unless a wall reflection could hide in between
        if 3 <= y3 <= 59 and 3 <= y2 <= 59:
            b1 = _ball_xy(stacks[i, 1])
            vy = (y3 - b1[1]) / 2.0 if (b1 and 3 <= b1[1] <= 59) else (y3 - y2)
        else:
            vy = y3 - y2
        vy = float(np.clip(vy, -E["max_vy"], E["max_vy"]))
        st["ball_x"][i], st["ball_y"][i] = x3, y3
        st["ball_vx"][i], st["ball_vy"][i] = vx, vy
        st["left_y"][i], st["right_y"][i] = ly, ry
    return st, keep


def make_env(st, keep):
    env = VecPong(int(keep.sum()), seed=999)
    env.reset()
    for k, v in st.items():
        getattr(env, k)[:] = v[keep]
    env.t[:] = 0
    return env


def lockstep(env, wm, stack0, mode, policy=None, acts=None, real=None, dev=None):
    n = stack0.shape[0]
    stack = stack0.clone()
    mse = np.zeros(H, np.float32)
    r_pred = np.zeros((n, H), np.float32)
    r_real = np.zeros((n, H), np.float32)
    hits = np.zeros((n, H), np.float32)
    alive = np.ones(n, bool)
    rng = np.random.default_rng(321)
    with torch.no_grad():
        for j in range(H):
            if mode == "policy":
                logits, _ = policy(stack)
                a_np = logits.argmax(dim=1).cpu().numpy()
            elif mode == "tracker":
                a_np = scripted_action(env.right_y, env.ball_y, 0.2, rng)
            else:                                        # logged
                a_np = acts[:, j]
            a = torch.from_numpy(np.asarray(a_np, np.int64)).to(dev)
            fl, r, _ = wm(stack, a)
            probs = torch.sigmoid(fl[:, 0])
            r_pred[:, j] = r.clamp(-1.5, 1.5).cpu().numpy()
            stack = torch.cat([stack[:, 1:], probs[:, None]], dim=1)
            if mode == "logged":
                f = real[:, j]                           # ground truth frames
            else:
                f, rr, d, info = env.step(a_np)
                r_real[:, j] = rr
                hits[:, j] = info["paddle_hit"]
            diff = (f.astype(np.float32) / 255 - probs.cpu().numpy()) ** 2
            per_env = diff.mean(axis=(1, 2))
            mse[j] = (per_env * alive).sum() / max(alive.sum(), 1)
            if mode != "logged":
                alive &= ~d
    return mse, r_pred, r_real, hits, alive


def main():
    config.seed_everything()
    dev = config.get_device()
    wm = load_wm(config.CKPT_DIR / "wm_v2.pt", dev, with_heads=True).eval()
    policy = load_policy(config.CKPT_DIR / "dream_agent.pt", dev).eval()
    data = TransitionData(config.DATA_DIR / "full")
    stacks, acts, real = data.val_windows(N, H)

    st, keep = reconstruct(stacks)
    n = int(keep.sum())
    stack0 = torch.from_numpy(
        stacks[keep].astype(np.float32) / 255.0).to(dev)

    # calibration: reconstructed env under LOGGED actions vs logged real frames
    env_c = make_env(st, keep)
    recon_mse = np.zeros(H, np.float32)
    f_cur = None
    for j in range(H):
        f_cur, _, _, _ = env_c.step(acts[keep][:, j])
        recon_mse[j] = ((f_cur.astype(np.float32) / 255
                         - real[keep][:, j].astype(np.float32) / 255) ** 2).mean()

    # stream 1: honest drift replication (WM + logged actions vs logged frames)
    honest_mse, _, _, _, _ = lockstep(None, wm, stack0, "logged",
                                      acts=acts[keep], real=real[keep], dev=dev)
    # stream 2: dream policy lockstep vs reconstructed env
    env_p = make_env(st, keep)
    pol_mse, pol_rpred, pol_rreal, pol_hits, alive_p = lockstep(
        env_p, wm, stack0, "policy", policy=policy, dev=dev)
    # stream 3: tracker (behavior policy) lockstep vs reconstructed env
    env_t = make_env(st, keep)
    trk_mse, trk_rpred, trk_rreal, trk_hits, _ = lockstep(
        env_t, wm, stack0, "tracker", dev=dev)

    ratio_pol = pol_mse / honest_mse
    ratio_trk = trk_mse / honest_mse
    over3 = np.flatnonzero(ratio_pol > 3.0)
    jstar = int(over3[0]) + 1 if len(over3) else None
    rp = pol_rpred.mean(0)

    def b(x, agg=np.sum):
        return {f"{a}-{c}": float(agg(x[a - 1:c])) for a, c in BINS}

    out = {
        "n_windows_kept": n, "n_dropped": int(N - n), "horizon": H,
        "reconstruction_floor_mse": recon_mse.tolist(),
        "honest_mse_logged_actions": honest_mse.tolist(),
        "policy_lockstep_mse": pol_mse.tolist(),
        "tracker_lockstep_mse": trk_mse.tolist(),
        "ratio_policy_vs_honest": ratio_pol.tolist(),
        "ratio_tracker_vs_honest": ratio_trk.tolist(),
        "binned": {
            "policy_mse": b(pol_mse, np.mean), "tracker_mse": b(trk_mse, np.mean),
            "honest_mse": b(honest_mse, np.mean), "recon_floor": b(recon_mse, np.mean),
            "ratio_policy": b(ratio_pol, np.mean), "ratio_tracker": b(ratio_trk, np.mean),
            "policy_imagined_reward": b(rp),
            "policy_real_reward": b(pol_rreal.mean(0)),
            "policy_real_hits": b(pol_hits.mean(0)),
            "tracker_imagined_reward": b(trk_rpred.mean(0)),
            "tracker_real_reward": b(trk_rreal.mean(0)),
            "tracker_real_hits": b(trk_hits.mean(0)),
        },
        "first_step_policy_ratio_gt3": jstar,
        "frac_imagined_reward_after_3x": (
            float(rp[jstar:].sum() / rp.sum()) if jstar and abs(rp.sum()) > 1e-9 else None),
        "totals": {
            "policy_imagined_per_dream": float(pol_rpred.sum(1).mean()),
            "policy_real_per_dream": float(pol_rreal.sum(1).mean()),
            "policy_real_hits_per_dream": float(pol_hits.sum(1).mean()),
            "tracker_imagined_per_dream": float(trk_rpred.sum(1).mean()),
            "tracker_real_per_dream": float(trk_rreal.sum(1).mean()),
            "tracker_real_hits_per_dream": float(trk_hits.sum(1).mean()),
        },
        "truncation_frac_imagined": {
            f"H{h}": float(rp[:h].sum() / rp.sum()) for h in (10, 15, 20)},
    }
    (config.RESULTS_DIR / "diag" / "horizon_valstate_lockstep.json").write_text(
        json.dumps(out, indent=2))
    print("VALSTATE_LOCKSTEP_OK kept=%d" % n)
    print("recon floor  @1,5,10,20,40:", [round(float(recon_mse[i]), 6) for i in (0, 4, 9, 19, 39)])
    print("honest (WM)  @1,5,10,20,40:", [round(float(honest_mse[i]), 6) for i in (0, 4, 9, 19, 39)])
    print("policy lock  @1,5,10,20,40:", [round(float(pol_mse[i]), 6) for i in (0, 4, 9, 19, 39)])
    print("tracker lock @1,5,10,20,40:", [round(float(trk_mse[i]), 6) for i in (0, 4, 9, 19, 39)])
    print(json.dumps({k: out[k] for k in ("binned", "first_step_policy_ratio_gt3",
                                          "frac_imagined_reward_after_3x", "totals",
                                          "truncation_frac_imagined")}, indent=2))


if __name__ == "__main__":
    main()
