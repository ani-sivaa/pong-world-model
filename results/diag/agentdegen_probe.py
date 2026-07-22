"""Probe: is the dream agent degenerate (parked / action-collapsed), or does it
track dreamed balls but not real ones?

(b) REAL env fingerprint: dream_agent vs ppo_baseline, VecPong(32, seed=123),
    300 steps, argmax. Uses ground-truth env state for positions.
(c) DREAM fingerprint: same policies driving wm.rollout.dream() from real val
    start stacks (horizon 40). Positions extracted from frames by masked
    centroid; the same extractor is validated on real binary frames.

Writes results/diag/agentdegen_results.json and prints a summary.
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
DZ = E["opp_deadzone"]

# ---------------------------------------------------------------- extractor
# frames: float [T,64,64] in [0,1] (probs or binary/255).
# right paddle lives in cols [60,62); ball anywhere else. Exclude walls
# (rows 0,63), left paddle cols [2,4), and a 1-col margin around paddles.
ROWS = np.arange(64)


def extract_positions(frames):
    """returns (pad_cy [T], ball_cy [T], ball_mass [T], pad_mass [T])"""
    f = np.asarray(frames, dtype=np.float64)
    inner = f[:, 1:63, :]                      # drop walls
    rows = ROWS[1:63][None, :]
    pad = inner[:, :, 60:62].sum(axis=2)       # [T,62]
    pad_mass = pad.sum(axis=1)
    pad_cy = (pad * rows).sum(axis=1) / np.maximum(pad_mass, 1e-6)
    ball = inner[:, :, 5:59].sum(axis=2)       # [T,62] excludes both paddles
    ball_mass = ball.sum(axis=1)
    ball_cy = (ball * rows).sum(axis=1) / np.maximum(ball_mass, 1e-6)
    return pad_cy, ball_cy, ball_mass, pad_mass


def fingerprint_from_traces(pad_cy, ball_cy, acts, valid=None):
    """pad_cy, ball_cy: [B,T] centers; acts: [B,T] ints; valid: [B,T] bool."""
    if valid is None:
        valid = np.ones_like(acts, dtype=bool)
    v = valid.astype(bool)
    n = max(int(v.sum()), 1)
    dist = np.abs(pad_cy - ball_cy)
    act_pct = [float((acts[v] == a).mean()) for a in range(3)]
    # paddle-y variance over time, per env, then mean over envs (valid steps)
    pvar = []
    for i in range(pad_cy.shape[0]):
        m = v[i]
        if m.sum() > 1:
            pvar.append(float(pad_cy[i][m].var()))
    diff = ball_cy - pad_cy                    # >0: ball below paddle -> down(2)
    off = v & (np.abs(diff) > DZ)
    toward = ((diff > 0) & (acts == 2)) | ((diff < 0) & (acts == 1))
    agree = float(toward[off].mean()) if off.sum() else float("nan")
    stay_off = float((acts[off] == 0).mean()) if off.sum() else float("nan")
    # correlation of paddle center with ball center across valid steps
    corr = float(np.corrcoef(pad_cy[v], ball_cy[v])[0, 1]) if n > 2 else float("nan")
    return {
        "pct_stay": act_pct[0], "pct_up": act_pct[1], "pct_down": act_pct[2],
        "paddle_y_var": float(np.mean(pvar)) if pvar else float("nan"),
        "mean_abs_track_err": float(dist[v].mean()),
        "median_abs_track_err": float(np.median(dist[v])),
        "toward_ball_when_offside": agree,
        "stay_when_offside": stay_off,
        "corr_paddle_ball_y": corr,
        "n_valid_steps": int(v.sum()),
    }


# ------------------------------------------------------------- (b) real env
@torch.no_grad()
def real_fingerprint(policy, n=32, steps=300, seed=123):
    env = VecPong(n, seed=seed)
    f = env.reset()
    stacks = np.repeat(f[:, None], config.FRAME_STACK, axis=1)
    pad_cy = np.zeros((n, steps)); ball_cy = np.zeros((n, steps))
    ball_vx = np.zeros((n, steps)); acts = np.zeros((n, steps), np.int64)
    hits = 0; pts_w = 0; pts_l = 0; frames_all = []
    for t in range(steps):
        x = torch.from_numpy(stacks).to(DEV).float() / 255.0
        logits, _ = policy(x)
        a = logits.argmax(dim=1).cpu().numpy()
        # ground-truth state BEFORE the step (what the policy is reacting to)
        pad_cy[:, t] = env.right_y + HALF_PH
        ball_cy[:, t] = env.ball_y + HALF_BS
        ball_vx[:, t] = env.ball_vx
        acts[:, t] = a
        f, r, d, info = env.step(a)
        hits += int(info["paddle_hit"].sum())
        pts_w += int((info["point"] == 1).sum())
        pts_l += int((info["point"] == -1).sum())
        frames_all.append(f.copy())
        stacks = np.concatenate([stacks[:, 1:], f[:, None]], axis=1)
        if d.any():
            stacks[d] = f[d][:, None]
    fp = fingerprint_from_traces(pad_cy, ball_cy, acts)
    inc = ball_vx > 0                          # ball moving toward agent
    fp_inc = fingerprint_from_traces(pad_cy, ball_cy, acts, valid=inc)
    fp["incoming_only"] = {k: fp_inc[k] for k in
                           ("mean_abs_track_err", "toward_ball_when_offside",
                            "pct_stay", "corr_paddle_ball_y")}
    fp["hits_per_env_300"] = hits / n
    fp["points"] = {"won": pts_w, "lost": pts_l}
    fp["mean_paddle_cy"] = float(pad_cy.mean())
    # extractor sanity: frame-based vs state-based positions on real frames
    fr = np.stack(frames_all, axis=1).astype(np.float64) / 255.0  # [n,steps,64,64]
    pcy_x, bcy_x, bm, pm = extract_positions(fr.reshape(-1, 64, 64))
    fp["extractor_check"] = {
        "pad_cy_mae": float(np.abs(pcy_x - pad_cy.reshape(-1)).mean()),
        # positions were recorded pre-step, frames are post-step -> compare
        # loosely; ball moves <=1.5px/step so MAE should be ~1px if extractor ok
        "ball_cy_mae": float(np.abs(bcy_x - ball_cy.reshape(-1)).mean()),
        "ball_mass_mean": float(bm.mean()), "pad_mass_mean": float(pm.mean()),
    }
    return fp


# --------------------------------------------------------------- (c) dreams
@torch.no_grad()
def dream_fingerprint(policy, wm, data, n=32, horizon=40, mode="argmax"):
    stacks, _, _ = data.val_windows(n, horizon)
    stack0 = torch.from_numpy(stacks.astype(np.float32) / 255.0).to(DEV)
    recorded = []

    def actions_fn(stack, j):
        logits, _ = policy(stack)
        if mode == "argmax":
            a = logits.argmax(dim=1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        return a

    frames, rewards, done_probs, actions = dream(wm, stack0, horizon, actions_fn)
    fr = frames.cpu().numpy()                  # [B,H,64,64] float probs
    acts = actions.cpu().numpy()
    B, H = fr.shape[:2]
    pcy, bcy, bmass, pmass = extract_positions(fr.reshape(-1, 64, 64))
    pcy = pcy.reshape(B, H); bcy = bcy.reshape(B, H)
    bmass = bmass.reshape(B, H); pmass = pmass.reshape(B, H)
    # a step is "valid" while the dream is alive and the ball is coherent
    alive = np.cumprod(np.concatenate(
        [np.ones((B, 1)), (1.0 - done_probs.cpu().numpy())[:, :-1]], axis=1), axis=1)
    valid = (alive > 0.5) & (bmass > 1.0)      # ball at least 1/4 rendered
    fp = fingerprint_from_traces(pcy, bcy, acts, valid=valid)
    rw = rewards.cpu().numpy()
    fp["dream_return_per_traj"] = float((rw * alive).sum(axis=1).mean())
    fp["pred_hits_per_traj"] = float(((rw > 0.05) & (rw < 0.5) & (alive > 0.5))
                                     .sum(axis=1).mean())
    fp["ball_mass_start"] = float(bmass[:, :5].mean())
    fp["ball_mass_end"] = float(bmass[:, -5:].mean())
    fp["pad_mass_mean"] = float(pmass.mean())
    fp["frac_steps_ball_coherent"] = float((bmass > 1.0).mean())
    fp["mean_alive_len"] = float(alive.sum(axis=1).mean())
    return fp


def main():
    torch.manual_seed(0)
    dream_pol = load_policy(config.CKPT_DIR / "dream_agent.pt", DEV).eval()
    base_pol = load_policy(config.CKPT_DIR / "ppo_baseline.pt", DEV).eval()
    wm = load_wm(config.CKPT_DIR / "wm_v2.pt", DEV, with_heads=True).eval()
    data = TransitionData(config.DATA_DIR / "full")

    out = {}
    for name, pol in [("dream_agent", dream_pol), ("ppo_baseline", base_pol)]:
        out[f"real::{name}"] = real_fingerprint(pol)
        print(f"real::{name}", json.dumps(out[f'real::{name}'], indent=1), flush=True)
    for name, pol in [("dream_agent", dream_pol), ("ppo_baseline", base_pol)]:
        out[f"dream::{name}::argmax"] = dream_fingerprint(pol, wm, data, mode="argmax")
        print(f"dream::{name}::argmax",
              json.dumps(out[f'dream::{name}::argmax'], indent=1), flush=True)
    out["dream::dream_agent::sampled"] = dream_fingerprint(
        dream_pol, wm, data, mode="sampled")
    print("dream::dream_agent::sampled",
          json.dumps(out["dream::dream_agent::sampled"], indent=1), flush=True)

    p = Path(__file__).parent / "agentdegen_results.json"
    p.write_text(json.dumps(out, indent=2))
    print("WROTE", p)


if __name__ == "__main__":
    main()
