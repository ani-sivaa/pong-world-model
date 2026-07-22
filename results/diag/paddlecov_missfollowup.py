"""paddlecov follow-up: joint-state test in the exact configuration the dream
policy creates in the real env — paddle pinned at the TOP (ry=1, action=UP
every step), ball approaching the right side heading DOWN (a guaranteed real
miss). Lockstep same-action WM-vs-real over 16 steps.

Case A ("dream-policy state"): paddle top, ball launched mid-court toward the
  bottom-right; actions all UP. Real outcome: ball sails past -> concede.
Case B (control, in-distribution): same ball launches, paddle pre-placed near
  the ball's arrival row; actions all STAY. Real outcome: paddle hit.

Measures per case: per-step lockstep MSE, dreamed vs real ball x, whether the
dreamed ball bounces back (hallucinated hit), predicted vs real cumulative
reward (reward head), done prob. Writes paddlecov_miss_results.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from pong.env import VecPong
from wm.model import load_wm
from wm.rollout import dream

E = config.ENV
DEV = config.get_device()
OUT = Path(__file__).parent / "paddlecov_miss_results.json"
H = 16
t0 = time.time()

# scenarios: ball starts at x=40 (right edge 42), moving right at +1.5
Y0 = [30.0, 35.0, 40.0, 45.0, 50.0]
VY = [0.0, 0.5, 1.0, 1.25]
scen = [(y0, vy) for y0 in Y0 for vy in VY]
S = len(scen)                                    # 20 per case
n = 2 * S

env = VecPong(n, seed=11)
env.reset()
y0s = np.array([s[0] for s in scen] * 2)
vys = np.array([s[1] for s in scen] * 2)
env.ball_x[:] = 40.0
env.ball_y[:] = y0s
env.ball_vx[:] = E["ball_vx"]
env.ball_vy[:] = vys
# case A (first S envs): paddle pinned top; case B: paddle placed at rough
# intercept row of the ball (13 steps ahead), so the real env gets a hit.
env.right_y[:S] = 1.0
arrive = np.clip(y0s[S:] + vys[S:] * 13.0, 1, 64 - 1 - E["paddle_h"] - 2)
env.right_y[S:] = np.clip(arrive - E["paddle_h"] / 2 + 1, 1.0, 51.0)
env.left_y[:] = 26.0

# actions: case A = UP (1) forever (paddle stays clamped at top, exactly what
# the dream agent does); case B = STAY (0).
acts_vec = np.concatenate([np.ones(S, np.int64), np.zeros(S, np.int64)])

# build a physically-consistent 4-stack by stepping 3 times (+ initial render)
hist = [env._render()]
for _ in range(3):
    f, _, _, _ = env.step(acts_vec)
    hist.append(f)
stack0_u8 = np.stack(hist[-4:], axis=1)                       # [n,4,64,64]

# ---- real lockstep continuation ----
real_frames = np.zeros((n, H, 64, 64), np.uint8)
real_rew = np.zeros((n, H), np.float32)
real_done = np.zeros((n, H), bool)
real_hit = np.zeros((n, H), bool)
done_seen = np.zeros(n, bool)
for j in range(H):
    f, r, d, info = env.step(acts_vec)
    tf = info["terminal_frame"]
    real_frames[:, j] = np.where(d[:, None, None], tf, f)
    real_rew[:, j] = np.where(done_seen, 0.0, r)
    real_hit[:, j] = info["paddle_hit"] & ~done_seen
    real_done[:, j] = d
    done_seen |= d

# ---- dreamed continuation, identical actions ----
wm = load_wm(config.CKPT_DIR / "wm_v2.pt", DEV, with_heads=True).eval()
stack0 = torch.from_numpy(stack0_u8.astype(np.float32) / 255.0).to(DEV)
acts_t = torch.from_numpy(acts_vec).to(DEV)
with torch.no_grad():
    dfr, drew, ddp, _ = dream(wm, stack0, H, lambda s, j: acts_t)
dfr_np = dfr.cpu().numpy()
drew_np = drew.cpu().numpy()
ddp_np = ddp.cpu().numpy()


def ball_com_x(fr, soft):
    """fr: [64,64] float [0,1] or uint8/255. Mask walls+paddle cols, COM x of
    remaining mass. Returns (x, mass)."""
    g = fr.astype(np.float64)
    if g.max() > 1.5:
        g = g / 255.0
    g = g.copy()
    g[0, :] = 0; g[63, :] = 0
    g[:, 2:4] = 0; g[:, 60:62] = 0
    if soft:
        g[g < 0.3] = 0.0
    else:
        g[g < 0.5] = 0.0
    m = g.sum()
    if m < 0.5:
        return np.nan, float(m)
    xs = np.arange(64, dtype=np.float64)
    return float((g.sum(axis=0) * xs).sum() / m), float(m)


rows = {"A_paddle_top_miss": [], "B_paddle_intercept_control": []}
per_step_mse = {"A": [], "B": []}
for i in range(n):
    case = "A" if i < S else "B"
    mse_curve = ((dfr_np[i] - real_frames[i].astype(np.float32) / 255.0) ** 2
                 ).mean(axis=(1, 2))
    per_step_mse[case].append(mse_curve)
    bx_real = [ball_com_x(real_frames[i, j], soft=False)[0] for j in range(H)]
    bx_dream = [ball_com_x(dfr_np[i, j], soft=True)[0] for j in range(H)]
    bxr = np.array(bx_real, np.float64)
    bxd = np.array(bx_dream, np.float64)
    # hallucinated bounce: dreamed ball reaches x>=55 then comes back below 50
    d_ok = ~np.isnan(bxd)
    halluc = False
    if d_ok.sum() >= 3:
        peaked = np.nanmax(bxd) >= 54.0
        after_peak = bxd[np.nanargmax(bxd):]
        halluc = bool(peaked and np.nanmin(after_peak) < 50.0)
    real_concede = bool((real_rew[i] < -0.5).any())
    real_hit_any = bool(real_hit[i].any())
    rows["A_paddle_top_miss" if case == "A" else "B_paddle_intercept_control"].append({
        "y0": float(y0s[i]), "vy": float(vys[i]),
        "real_concede": real_concede, "real_hit": real_hit_any,
        "real_cum_reward": float(real_rew[i].sum()),
        "pred_cum_reward": float(drew_np[i].sum()),
        "pred_rhit_steps": int((drew_np[i] > 0.05).sum()),
        "dream_ball_bounced_back": halluc,
        "dream_ball_vanished": bool(np.isnan(bxd[-4:]).all()),
        "mse_at_step8": float(mse_curve[7]), "mse_at_step16": float(mse_curve[15]),
        "max_done_prob": float(ddp_np[i].max()),
        "ball_x_real_last_visible": float(np.nanmax(bxr)) if not np.isnan(bxr).all() else None,
    })

summary = {}
for case, key in (("A", "A_paddle_top_miss"), ("B", "B_paddle_intercept_control")):
    rr = rows[key]
    mc = np.stack(per_step_mse[case]).mean(axis=0)
    summary[key] = {
        "n": len(rr),
        "real_concede_rate": float(np.mean([r["real_concede"] for r in rr])),
        "real_hit_rate": float(np.mean([r["real_hit"] for r in rr])),
        "mean_real_cum_reward": float(np.mean([r["real_cum_reward"] for r in rr])),
        "mean_pred_cum_reward": float(np.mean([r["pred_cum_reward"] for r in rr])),
        "dream_bounce_back_rate": float(np.mean([r["dream_ball_bounced_back"] for r in rr])),
        "dream_ball_vanished_rate": float(np.mean([r["dream_ball_vanished"] for r in rr])),
        "mean_max_done_prob": float(np.mean([r["max_done_prob"] for r in rr])),
        "mse_curve_mean": mc.tolist(),
    }
out = {"summary": summary, "rows": rows}
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps(summary["A_paddle_top_miss"], indent=1))
print(json.dumps(summary["B_paddle_intercept_control"], indent=1))
print(f"[{time.time()-t0:.1f}s] wrote {OUT}")
