"""rewardhal probe: does the WM reward head hallucinate paddle-hit rewards?

(a) On-distribution calibration: val transitions, reward-head pred vs true reward.
(b) Off-distribution: dreams driven by the dream policy (argmax); for every
    predicted-reward event (pred > 0.05) check pixel-level physical plausibility
    (ball actually near the right paddle) in the dreamed frame itself.

Read-only w.r.t. repo code/checkpoints/data. Writes JSON to results/diag/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from wm.data import TransitionData
from wm.model import load_wm
from wm.rollout import dream
from agents.policy import load_policy

t0 = time.time()
config.seed_everything()
dev = config.get_device()
print(f"device={dev}")

wm = load_wm(config.CKPT_DIR / "wm_v2.pt", dev, with_heads=True).eval()
pol = load_policy(config.CKPT_DIR / "dream_agent.pt", dev).eval()
data = TransitionData(config.DATA_DIR / "full")

R_HIT = config.ENV["r_hit"]          # 0.1
THRESH = 0.05

# ------------------------------------------------------------------ (a) ----
N_CAL = 4096   # ~2x the asked 2000 so rare reward events have decent counts
stack, acts, targets, rews, dns, alive = data.sample(N_CAL, 1, val=True)
r_true = rews[:, 0]

preds = np.zeros(N_CAL, dtype=np.float32)
with torch.no_grad():
    for i in range(0, N_CAL, 512):
        s = torch.from_numpy(stack[i:i + 512].astype(np.float32) / 255.0).to(dev)
        a = torch.from_numpy(acts[i:i + 512, 0]).to(dev)
        _, r, _ = wm(s, a)
        preds[i:i + 512] = r.cpu().numpy()

mse = float(np.mean((preds - r_true) ** 2))

def cls(r):
    if abs(r - R_HIT) < 1e-4: return "hit(+0.1)"
    if r > 0.5: return "score(+1)"
    if r < -0.5: return "concede(-1)"
    return "zero"

classes = np.array([cls(r) for r in r_true])
per_class = {}
for c in ["zero", "hit(+0.1)", "score(+1)", "concede(-1)"]:
    m = classes == c
    if m.sum() == 0:
        per_class[c] = dict(n=0)
        continue
    p = preds[m]
    per_class[c] = dict(
        n=int(m.sum()),
        pred_mean=float(p.mean()), pred_std=float(p.std()),
        pred_min=float(p.min()), pred_max=float(p.max()),
        frac_pred_gt_thresh=float((p > THRESH).mean()),
    )

# precision / recall for r_hit detection at pred > 0.05
true_pos_mask = classes == "hit(+0.1)"
pred_pos = preds > THRESH
tp = int((pred_pos & true_pos_mask).sum())
fp = int((pred_pos & ~true_pos_mask).sum())
fn = int((~pred_pos & true_pos_mask).sum())
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
# variant: exclude true score(+1) events from the FP set (they legitimately
# push pred above 0.05) so precision reflects hit-vs-nothing confusion
fp_excl_score = int((pred_pos & (classes == "zero")).sum() +
                    (pred_pos & (classes == "concede(-1)")).sum())
precision_excl_score = tp / max(tp + fp_excl_score, 1)

cal = dict(n=N_CAL, mse=mse, threshold=THRESH,
           precision=precision, recall=recall,
           precision_excluding_true_score_events=precision_excl_score,
           tp=tp, fp=fp, fn=fn, per_class=per_class)
print("=== (a) on-distribution calibration ===")
print(json.dumps(cal, indent=2))

# ------------------------------------------------------------------ (b) ----
N_DREAMS, H = 64, config.DREAM["horizon"]
stacks, _, _ = data.val_windows(N_DREAMS, H)
stack0 = torch.from_numpy(stacks.astype(np.float32) / 255.0).to(dev)

def policy_argmax(s, j):
    with torch.no_grad():
        logits, _ = pol(s)
    return logits.argmax(dim=-1)

frames, rewards, done_probs, actions = dream(wm, stack0, H, policy_argmax)
frames_np = frames.cpu().numpy()          # [B,H,64,64] float probs
rewards_np = rewards.cpu().numpy()        # [B,H]

# ---- pixel-based physical plausibility checker ----------------------------
# env geometry: right paddle cols [60,62); left paddle cols [2,4); walls rows 0,63
PAD_COLS = (60, 61)
LEFT_PAD_COLS = (2, 3)

def analyze_frame(fr):
    """fr: [64,64] float probs. Returns (paddle_rows, ball_pixels rows/cols)."""
    b = fr > 0.5
    b[0, :] = False
    b[63, :] = False
    pad = b[:, PAD_COLS[0]] | b[:, PAD_COLS[1]]
    paddle_rows = np.flatnonzero(pad)
    ball = b.copy()
    ball[:, [PAD_COLS[0], PAD_COLS[1], LEFT_PAD_COLS[0], LEFT_PAD_COLS[1]]] = False
    br, bc = np.nonzero(ball)
    return paddle_rows, br, bc

def ball_near_paddle(fr, x_margin, y_margin):
    """True if some ball pixel is within x_margin cols of paddle col 60 and
    within y_margin rows of the paddle's row span."""
    paddle_rows, br, bc = analyze_frame(fr)
    if len(br) == 0 or len(paddle_rows) == 0:
        return False, len(br), (int(bc.max()) if len(bc) else -1)
    near_x = bc >= (PAD_COLS[0] - x_margin)          # right edge of ball close to col 60
    if not near_x.any():
        return False, len(br), int(bc.max())
    lo, hi = paddle_rows.min() - y_margin, paddle_rows.max() + y_margin
    ok = near_x & (br >= lo) & (br <= hi)
    return bool(ok.any()), len(br), int(bc.max())

events = []           # every (dream, step) with pred reward > THRESH
for b in range(N_DREAMS):
    for j in range(H):
        r = float(rewards_np[b, j])
        if r > THRESH:
            fr = frames_np[b, j]
            strict, n_ball, max_col = ball_near_paddle(fr, x_margin=2, y_margin=1)
            lenient, _, _ = ball_near_paddle(fr, x_margin=4, y_margin=3)
            # also check the previous dreamed frame (contact could render 1 step earlier)
            prev = frames_np[b, j - 1] if j > 0 else stack0[b, -1].cpu().numpy()
            lenient_prev, _, _ = ball_near_paddle(prev, x_margin=4, y_margin=3)
            events.append(dict(dream=b, step=j, pred=r, n_ball_px=n_ball,
                               ball_max_col=max_col, strict=strict,
                               lenient=lenient,
                               lenient_either=(lenient or lenient_prev)))

n_ev = len(events)
sum_per_dream = rewards_np.sum(axis=1)
hitlike = [e for e in events if e["pred"] <= 0.5]
scorelike = [e for e in events if e["pred"] > 0.5]

def frac(evs, key, invert=True):
    if not evs: return None
    v = np.mean([not e[key] if invert else e[key] for e in evs])
    return float(v)

dreams = dict(
    n_dreams=N_DREAMS, horizon=H,
    mean_pred_reward_sum_per_dream=float(sum_per_dream.mean()),
    std_pred_reward_sum_per_dream=float(sum_per_dream.std()),
    n_events_pred_gt_thresh=n_ev,
    events_per_dream=n_ev / N_DREAMS,
    n_hitlike_events=len(hitlike), n_scorelike_events=len(scorelike),
    frac_events_NO_ball_near_paddle_strict=frac(events, "strict"),
    frac_events_NO_ball_near_paddle_lenient=frac(events, "lenient"),
    frac_events_NO_ball_near_paddle_lenient_either_frame=frac(events, "lenient_either"),
    frac_hitlike_NO_ball_near_paddle_lenient=frac(hitlike, "lenient"),
    frac_events_with_zero_ball_pixels=float(np.mean([e["n_ball_px"] == 0 for e in events])) if n_ev else None,
    ball_max_col_at_events_median=float(np.median([e["ball_max_col"] for e in events])) if n_ev else None,
    ball_max_col_histogram={},
)
if n_ev:
    cols = np.array([e["ball_max_col"] for e in events])
    bins = {"no_ball(-1)": int((cols == -1).sum()),
            "left_half(<32)": int(((cols >= 0) & (cols < 32)).sum()),
            "mid(32-55)": int(((cols >= 32) & (cols <= 55)).sum()),
            "near_paddle(56-59)": int(((cols >= 56) & (cols <= 59)).sum()),
            "at/behind(>=60)": int((cols >= 60).sum())}
    dreams["ball_max_col_histogram"] = bins

# sanity check: same checker on REAL frames at true hit events (validates the checker)
chk_stack, chk_acts, chk_tgts, chk_rews, _, _ = data.sample(8192, 1, val=True)
hit_idx = np.flatnonzero(np.abs(chk_rews[:, 0] - R_HIT) < 1e-4)[:200]
checker_ok = []
for i in hit_idx:
    fr = chk_tgts[i, 0].astype(np.float32) / 255.0    # real next frame at hit
    ok, _, _ = ball_near_paddle(fr, x_margin=4, y_margin=3)
    checker_ok.append(ok)
dreams["checker_sanity_frac_true_hits_flagged_near_paddle"] = (
    float(np.mean(checker_ok)) if checker_ok else None)
dreams["checker_sanity_n_true_hits"] = len(checker_ok)

print("=== (b) dream-policy off-distribution ===")
print(json.dumps(dreams, indent=2))

# a few example hallucinated events for the report
bad = [e for e in events if not e["lenient"]][:10]
print("example implausible events:", json.dumps(bad, indent=2))

out = dict(calibration=cal, dreams=dreams,
           example_implausible_events=bad,
           runtime_s=time.time() - t0)
Path(config.RESULTS_DIR / "diag" / "rewardhal_results.json").write_text(
    json.dumps(out, indent=2))
print(f"DONE in {time.time() - t0:.1f}s")
