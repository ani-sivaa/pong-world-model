"""Probe 2: in dreams, who closes the paddle-ball gap — the paddle (agent
tracking) or the ball (WM hallucinating the ball toward the paddle)?
Also: geometry of WM-predicted hit rewards (is r_hit paid without contact
geometry?), and dream paddle resting position.
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

DEV = config.get_device()
E = config.ENV
HALF_PH = E["paddle_h"] / 2.0
HALF_BS = E["ball_size"] / 2.0


def extract_xy(frames):
    """frames float [T,64,64] -> pad_cy, ball_cy, ball_cx, ball_mass."""
    f = np.asarray(frames, dtype=np.float64)
    inner = f[:, 1:63, :]
    rows = np.arange(1, 63)[None, :]
    cols = np.arange(5, 59)[None, :]
    pad = inner[:, :, 60:62].sum(axis=2)
    pad_cy = (pad * rows).sum(1) / np.maximum(pad.sum(1), 1e-6)
    ballr = inner[:, :, 5:59]
    bm = ballr.sum(axis=(1, 2))
    ball_cy = ((ballr.sum(2)) * rows).sum(1) / np.maximum(bm, 1e-6)
    ball_cx = ((ballr.sum(1)) * cols).sum(1) / np.maximum(bm, 1e-6)
    return pad_cy, ball_cy, ball_cx, bm


def attribution(pad_cy, ball_cy, valid):
    """Per consecutive valid step pair: gap change split into paddle part and
    ball part. gap = ball_cy - pad_cy. paddle contribution = -(dpad) * sign(gap)
    (paddle moving toward ball shrinks gap), ball contribution = dball*sign(gap).
    Returns mean signed contributions (negative = closes the gap) and
    P(ball step moves toward paddle)."""
    B, T = pad_cy.shape
    pc, bc, ptow, btow = [], [], [], []
    for i in range(B):
        for t in range(T - 1):
            if not (valid[i, t] and valid[i, t + 1]):
                continue
            gap = ball_cy[i, t] - pad_cy[i, t]
            if abs(gap) < 1.0:
                continue
            s = np.sign(gap)
            dpad = pad_cy[i, t + 1] - pad_cy[i, t]
            dball = ball_cy[i, t + 1] - ball_cy[i, t]
            pc.append(-dpad * s)     # <0 means paddle closed the gap
            bc.append(dball * -s)    # <0 means ball closed the gap... wait
            # keep consistent: contribution to gap change:
            # d|gap| ~ s*(dball - dpad). ball part = s*dball, paddle part = -s*dpad
            ptow.append(-s * dpad < -0.25)   # paddle stepped toward ball
            btow.append(s * dball < -0.25)   # ball stepped toward paddle
    pc = np.array(pc); bc = np.array(bc)
    return {
        "mean_gapchange_from_paddle": float(np.mean(-np.array(pc))) if len(pc) else None,
        "p_paddle_steps_toward_ball": float(np.mean(ptow)),
        "p_ball_steps_toward_paddle": float(np.mean(btow)),
        "n_pairs": len(ptow),
    }


@torch.no_grad()
def run_dream(policy, wm, data, n=32, horizon=40):
    stacks, _, _ = data.val_windows(n, horizon)
    stack0 = torch.from_numpy(stacks.astype(np.float32) / 255.0).to(DEV)
    frames, rewards, done_probs, actions = dream(
        wm, stack0, horizon, lambda s, j: policy(s)[0].argmax(dim=1))
    fr = frames.cpu().numpy(); rw = rewards.cpu().numpy()
    B, H = fr.shape[:2]
    pcy, bcy, bcx, bm = extract_xy(fr.reshape(-1, 64, 64))
    pcy, bcy = pcy.reshape(B, H), bcy.reshape(B, H)
    bcx, bm = bcx.reshape(B, H), bm.reshape(B, H)
    alive = np.cumprod(np.concatenate(
        [np.ones((B, 1)), (1.0 - done_probs.cpu().numpy())[:, :-1]], 1), 1)
    valid = (alive > 0.5) & (bm > 1.0)
    res = attribution(pcy, bcy, valid)
    res["mean_pad_cy_dream"] = float(pcy[valid].mean())
    res["mean_ball_cy_dream"] = float(bcy[valid].mean())
    res["mean_ball_cx_dream"] = float(bcx[valid].mean())
    # predicted-hit geometry: steps with r_hit-like predicted reward
    hit_mask = (rw > 0.05) & (rw < 0.5) & (alive > 0.5)
    if hit_mask.any():
        gy = np.abs(bcy - pcy)[hit_mask]
        gx = bcx[hit_mask]
        res["pred_hit_steps"] = {
            "count": int(hit_mask.sum()),
            "ball_cx_mean": float(gx.mean()), "ball_cx_min": float(gx.min()),
            "frac_ball_cx_gt_50": float((gx > 50).mean()),
            "abs_dy_mean": float(gy.mean()),
            "frac_geom_plausible": float(((gx > 50) & (gy < 8)).mean()),
        }
    # reward summary
    res["sum_pred_reward"] = float((rw * alive).sum(1).mean())
    res["mean_pred_reward_per_step"] = float(rw[alive > 0.5].mean())
    return res


def real_ball_control(n=32, steps=300, seed=123):
    """Control: in the real env under the dream agent, P(ball y-step moves
    toward paddle) — physics baseline for the attribution numbers."""
    policy = load_policy(config.CKPT_DIR / "dream_agent.pt", DEV).eval()
    env = VecPong(n, seed=seed)
    f = env.reset()
    stacks = np.repeat(f[:, None], config.FRAME_STACK, axis=1)
    pcy = np.zeros((n, steps)); bcy = np.zeros((n, steps))
    dn = np.zeros((n, steps), bool)
    for t in range(steps):
        x = torch.from_numpy(stacks).to(DEV).float() / 255.0
        with torch.no_grad():
            a = policy(x)[0].argmax(dim=1).cpu().numpy()
        pcy[:, t] = env.right_y + HALF_PH
        bcy[:, t] = env.ball_y + HALF_BS
        f, r, d, info = env.step(a)
        dn[:, t] = d
        stacks = np.concatenate([stacks[:, 1:], f[:, None]], axis=1)
        if d.any():
            stacks[d] = f[d][:, None]
    valid = ~dn  # exclude reset steps
    return attribution(pcy, bcy, valid)


def main():
    torch.manual_seed(0)
    wm = load_wm(config.CKPT_DIR / "wm_v2.pt", DEV, with_heads=True).eval()
    data = TransitionData(config.DATA_DIR / "full")
    out = {}
    for name in ["dream_agent", "ppo_baseline"]:
        pol = load_policy(config.CKPT_DIR / f"{name}.pt", DEV).eval()
        out[f"dream::{name}"] = run_dream(pol, wm, data)
        print(f"dream::{name}", json.dumps(out[f'dream::{name}'], indent=1), flush=True)
    out["real::dream_agent::physics_control"] = real_ball_control()
    print("real control", json.dumps(out["real::dream_agent::physics_control"], indent=1))
    p = Path(__file__).parent / "agentdegen_results2.json"
    p.write_text(json.dumps(out, indent=2))
    print("WROTE", p)


if __name__ == "__main__":
    main()
