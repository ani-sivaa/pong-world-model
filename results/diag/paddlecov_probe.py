"""paddlecov probe: does the dream policy drive the RIGHT paddle into
state regions the training data never covered, and does WM error explode there?

(a) histogram right-paddle top-row y over ~200k training frames (data/full)
(b) paddle-y histograms for the dream policy in the REAL env and in DREAMS;
    histogram intersections vs the data distribution
(c) WM 1-step MSE on val transitions grouped by paddle-y bin (real stacks +
    logged actions), per-bin (visit_freq, mse)
(c+) controlled teleport lockstep: paddle teleported to each y, stack built
    physically (4 steps of action 0), WM 1-step vs real 1-step, per-y MSE.
    Removes the "rare bins in val data" confound.

Writes results/diag/paddlecov_results.json. Read-only w.r.t. code/ckpts/data.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from agents.policy import load_policy
from pong.env import VecPong, _round_px
from wm.data import TransitionData
from wm.model import load_wm
from wm.rollout import dream

E = config.ENV
RY_MIN, RY_MAX = 1, config.ENV["H"] - 1 - E["paddle_h"]  # 1 .. 51 paddle top y
N_VALS = RY_MAX - RY_MIN + 1                              # 51
COARSE_EDGES = np.array([1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 52])  # 10 bins
DEV = config.get_device()
OUT = Path(__file__).parent / "paddlecov_results.json"
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- extractors
def paddle_y_strict(frames_u8):
    """frames_u8: uint8 [N,64,64]. Right paddle top y via cols 60:62.
    Strict: exactly paddle_h contiguous white rows (ball-overlap frames -> -1).
    Returns int array [N], -1 where ambiguous."""
    patch = frames_u8[:, 1:63, E["right_x"]:E["right_x"] + E["paddle_w"]]
    mask = (patch == 255).any(axis=2)                    # [N,62] rows 1..62
    cnt = mask.sum(axis=1)
    first = mask.argmax(axis=1)                          # 0-based within rows 1..62
    last = 61 - mask[:, ::-1].argmax(axis=1)
    ok = (cnt == E["paddle_h"]) & ((last - first + 1) == E["paddle_h"])
    out = np.where(ok, first + 1, -1)
    return out.astype(np.int64)


def paddle_y_soft(frames_f, thresh=0.5, min_mass=6.0):
    """frames_f: float [N,64,64] in [0,1] (dreamed probs). COM-based estimate.
    Returns (ry int [N] with -1 for missing paddle, missing_frac)."""
    patch = frames_f[:, 1:63, E["right_x"]:E["right_x"] + E["paddle_w"]].mean(axis=2)
    w = np.where(patch > thresh, patch, 0.0)             # [N,62]
    mass = w.sum(axis=1)
    rows = np.arange(1, 63, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cy = (w * rows).sum(axis=1) / np.maximum(mass, 1e-9)
    ry = np.clip(np.floor(cy - (E["paddle_h"] - 1) / 2.0 + 0.5), RY_MIN, RY_MAX)
    ok = mass >= min_mass * thresh                        # enough paddle pixels
    out = np.where(ok, ry, -1).astype(np.int64)
    return out, float(1.0 - ok.mean())


def hist51(ry):
    """Normalized histogram over ry values 1..51 (drops -1)."""
    ry = ry[ry >= 0]
    h = np.bincount(ry - RY_MIN, minlength=N_VALS).astype(np.float64)
    return h / max(h.sum(), 1.0)


def intersection(p, q):
    return float(np.minimum(p, q).sum())


def spearman(x, y):
    """Spearman rho via Pearson on average ranks (numpy only)."""
    def rank(a):
        order = np.argsort(a, kind="stable")
        r = np.empty(len(a), np.float64)
        r[order] = np.arange(len(a), dtype=np.float64)
        # average ties
        for v in np.unique(a):
            m = a == v
            r[m] = r[m].mean()
        return r
    rx, ry_ = rank(np.asarray(x, np.float64)), rank(np.asarray(y, np.float64))
    rx -= rx.mean(); ry_ -= ry_.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry_ ** 2).sum())
    return float((rx * ry_).sum() / denom) if denom > 0 else 0.0


def coarse(ry):
    """Coarse-bin normalized histogram (10 bins) for readability."""
    ry = ry[ry >= 0]
    h, _ = np.histogram(ry, bins=COARSE_EDGES)
    return (h / max(h.sum(), 1)).tolist()


results = {"coarse_bin_edges": COARSE_EDGES.tolist()}

# ============================== (a) data histogram ===========================
log("(a) data/full paddle-y histogram over ~200k train frames")
data = TransitionData(config.DATA_DIR / "full")
frames = data.frames
stride_idx_end = data.val_begin                          # train region only
ry_data_chunks = []
n_amb = n_tot = 0
for c0 in range(0, stride_idx_end, 50_000):
    sub = np.asarray(frames[c0:min(c0 + 50_000, stride_idx_end):5])
    ry = paddle_y_strict(sub)
    n_amb += int((ry < 0).sum()); n_tot += len(ry)
    ry_data_chunks.append(ry[ry >= 0])
ry_data = np.concatenate(ry_data_chunks)
h_data = hist51(ry_data)
results["a_data"] = {
    "n_frames_scanned": n_tot, "n_used": int(len(ry_data)),
    "ambiguous_frac": n_amb / n_tot,
    "hist51": h_data.tolist(), "coarse_hist": coarse(ry_data),
    "mean_ry": float(ry_data.mean()), "std_ry": float(ry_data.std()),
    "frac_ry_le_5": float((ry_data <= 5).mean()),
    "frac_ry_eq_1": float((ry_data == 1).mean()),
    "frac_ry_ge_47": float((ry_data >= 47).mean()),
}
log(f"    data mean ry={ry_data.mean():.1f} frac(ry<=5)={results['a_data']['frac_ry_le_5']:.4f} "
    f"frac(ry==1)={results['a_data']['frac_ry_eq_1']:.5f}")

# ============================== (b) policy rollouts ==========================
log("(b) dream policy in REAL env (VecPong(32) x 40 steps x 4 seeds, argmax)")
policies = {
    "dream_agent": load_policy(config.CKPT_DIR / "dream_agent.pt", DEV).eval(),
    "ppo_baseline": load_policy(config.CKPT_DIR / "ppo_baseline.pt", DEV).eval(),
}
real_ry, real_actions, real_dones = {}, {}, {}
with torch.no_grad():
    for name, pol in policies.items():
        rys, acts_all, dones_n = [], [], 0
        for seed in (123, 124, 125, 126):
            env = VecPong(32, seed=seed)
            f = env.reset()
            stacks = np.repeat(f[:, None], config.FRAME_STACK, axis=1)
            for _ in range(40):
                x = torch.from_numpy(stacks).to(DEV).float() / 255.0
                logits, _ = pol(x)
                a = logits.argmax(dim=1).cpu().numpy()
                f, r, d, info = env.step(a)
                rys.append(_round_px(env.right_y).copy())
                acts_all.append(a)
                dones_n += int(d.sum())
                stacks = np.concatenate([stacks[:, 1:], f[:, None]], axis=1)
                if d.any():
                    stacks[d] = f[d][:, None]
        real_ry[name] = np.concatenate(rys)
        real_actions[name] = np.concatenate(acts_all)
        real_dones[name] = dones_n

log("(b) dream policy in DREAMS (256 real starts x horizon 40, argmax)")
wm = load_wm(config.CKPT_DIR / "wm_v2.pt", DEV, with_heads=True).eval()
dream_ry, dream_missing = {}, {}
with torch.no_grad():
    stacks_u8, *_ = data.sample(256, 1)                  # real starts, train region
    stack0 = torch.from_numpy(stacks_u8.astype(np.float32) / 255.0).to(DEV)
    for name, pol in policies.items():
        dfr, drew, ddp, dact = dream(
            wm, stack0, config.DREAM["horizon"],
            lambda s, j, p=pol: p(s)[0].argmax(dim=1))
        dfr_np = dfr.cpu().numpy().reshape(-1, 64, 64)   # [256*40,64,64]
        ry, miss = paddle_y_soft(dfr_np)
        dream_ry[name] = ry
        dream_missing[name] = miss
        if name == "dream_agent":
            results["b_dream_agent_dream_stats"] = {
                "imagined_return_per_traj": float(drew.sum(1).mean()),
                "imagined_hits_per_traj": float((drew > 0.05).float().sum(1).mean()),
                "mean_done_prob_final": float(ddp[:, -1].mean()),
            }

h = {"data": h_data}
for name in policies:
    h[f"real_{name}"] = hist51(real_ry[name])
    h[f"dream_{name}"] = hist51(dream_ry[name])

results["b_hists"] = {}
for name in policies:
    a_frac = np.bincount(real_actions[name], minlength=3) / len(real_actions[name])
    rr = real_ry[name]; dr = dream_ry[name][dream_ry[name] >= 0]
    results["b_hists"][name] = {
        "real_n": int(len(rr)), "real_mean_ry": float(rr.mean()),
        "real_frac_ry_le_5": float((rr <= 5).mean()),
        "real_frac_ry_eq_1": float((rr == 1).mean()),
        "real_action_frac_stay_up_down": a_frac.tolist(),
        "real_dones_seen": real_dones[name],
        "real_coarse_hist": coarse(rr),
        "dream_n_used": int(len(dr)), "dream_missing_frac": dream_missing[name],
        "dream_mean_ry": float(dr.mean()),
        "dream_frac_ry_le_5": float((dr <= 5).mean()),
        "dream_coarse_hist": coarse(dream_ry[name]),
    }
results["b_overlap"] = {
    "data_vs_real_dream_agent": intersection(h["data"], h["real_dream_agent"]),
    "data_vs_dream_dream_agent": intersection(h["data"], h["dream_dream_agent"]),
    "real_vs_dream_dream_agent": intersection(h["real_dream_agent"], h["dream_dream_agent"]),
    "data_vs_real_ppo_baseline": intersection(h["data"], h["real_ppo_baseline"]),
    "data_vs_dream_ppo_baseline": intersection(h["data"], h["dream_ppo_baseline"]),
}
log(f"    overlaps: {json.dumps(results['b_overlap'])}")

# ============================== (c) conditioned val error ====================
log("(c) WM 1-step MSE on ALL val transitions, grouped by paddle-y bin")
t_val = data._valid(data.val_begin, data.T, 1)
ry_val = paddle_y_strict(np.asarray(frames[t_val]))      # paddle y at stack end
keep = ry_val >= 0
t_val, ry_val = t_val[keep], ry_val[keep]
mses = np.zeros(len(t_val), np.float32)
mses_strip = np.zeros(len(t_val), np.float32)            # cols 52:64 (paddle region)
with torch.no_grad():
    for i0 in range(0, len(t_val), 512):
        tt = t_val[i0:i0 + 512]
        st = np.asarray(frames[tt[:, None] + np.arange(-3, 1)], np.float32) / 255.0
        tg = np.asarray(frames[tt + 1], np.float32) / 255.0
        ac = torch.from_numpy(data.actions[tt].astype(np.int64)).to(DEV)
        st_t = torch.from_numpy(st).to(DEV)
        logits, _, _ = wm(st_t, ac)
        pred = torch.sigmoid(logits[:, 0]).cpu().numpy()
        d2 = (pred - tg) ** 2
        mses[i0:i0 + len(tt)] = d2.mean(axis=(1, 2))
        mses_strip[i0:i0 + len(tt)] = d2[:, :, 52:64].mean(axis=(1, 2))

bin_idx = np.digitize(ry_val, COARSE_EDGES) - 1          # 0..9
c_rows = []
for b in range(len(COARSE_EDGES) - 1):
    m = bin_idx == b
    c_rows.append({
        "ry_range": [int(COARSE_EDGES[b]), int(COARSE_EDGES[b + 1] - 1)],
        "n_val": int(m.sum()),
        "visit_freq_val": float(m.mean()),
        "visit_freq_traindata": float(h_data[max(0, COARSE_EDGES[b] - 1):COARSE_EDGES[b + 1] - 1].sum()),
        "mse_1step": float(mses[m].mean()) if m.any() else None,
        "mse_1step_rightstrip": float(mses_strip[m].mean()) if m.any() else None,
    })
results["c_val_by_bin"] = c_rows
results["c_val_total"] = {"n": int(len(t_val)), "mse_overall": float(mses.mean())}
freq = np.array([r["visit_freq_val"] for r in c_rows])
mm = np.array([r["mse_1step"] if r["mse_1step"] is not None else np.nan for r in c_rows])
ok = ~np.isnan(mm) & (freq > 0)
if ok.sum() > 2:
    results["c_spearman_visitfreq_vs_mse"] = {"rho": spearman(freq[ok], mm[ok])}
log(f"    per-bin: {json.dumps(c_rows)}")

# ============================== (c+) teleport lockstep =======================
log("(c+) controlled teleport lockstep: paddle forced to each y, WM vs real 1-step")
ry_grid = np.arange(RY_MIN, RY_MAX + 1, 2)               # 26 values
per_y = 24
n = len(ry_grid) * per_y
env = VecPong(n, seed=7)
env.reset()
forced = np.repeat(ry_grid, per_y).astype(np.float64)
env.right_y[:] = forced
hist_frames = [env._render()]
for _ in range(3):                                        # action 0 -> paddle static
    f, _, _, _ = env.step(np.zeros(n, np.int64))
    hist_frames.append(f)
stack_np = np.stack(hist_frames[-4:], axis=1).astype(np.float32) / 255.0  # [n,4,64,64]
test_actions = (np.arange(n) % 3).astype(np.int64)        # cycle stay/up/down
f_real, _, d_real, _ = env.step(test_actions)             # real next frame
assert not d_real.any()
with torch.no_grad():
    logits, _, _ = wm(torch.from_numpy(stack_np).to(DEV),
                      torch.from_numpy(test_actions).to(DEV))
    pred = torch.sigmoid(logits[:, 0]).cpu().numpy()
tgt = f_real.astype(np.float32) / 255.0
d2 = (pred - tgt) ** 2
mse_all = d2.mean(axis=(1, 2))
mse_strip = d2[:, :, 52:64].mean(axis=(1, 2))
# also: predicted-paddle position error (does WM even keep the paddle there?)
pred_ry, pred_miss = paddle_y_soft(pred)
true_ry = paddle_y_strict(f_real)
rows = []
for i, ry in enumerate(ry_grid):
    m = slice(i * per_y, (i + 1) * per_y)
    pr, tr = pred_ry[m], true_ry[m]
    ok2 = (pr >= 0) & (tr >= 0)
    rows.append({
        "ry": int(ry),
        "traindata_freq": float(h_data[ry - RY_MIN]),
        "mse_1step": float(mse_all[m].mean()),
        "mse_1step_rightstrip": float(mse_strip[m].mean()),
        "paddle_pos_abs_err": float(np.abs(pr[ok2] - tr[ok2]).mean()) if ok2.any() else None,
        "paddle_missing_frac": float((pred_ry[m] < 0).mean()),
    })
results["cplus_teleport_by_ry"] = rows
tf = np.array([r["traindata_freq"] for r in rows])
tm = np.array([r["mse_1step"] for r in rows])
results["cplus_spearman_trainfreq_vs_mse"] = {"rho": spearman(tf, tm)}
rare = tf < 0.005
results["cplus_summary"] = {
    "mse_rare_bins(train_freq<0.005)": float(tm[rare].mean()) if rare.any() else None,
    "mse_common_bins": float(tm[~rare].mean()) if (~rare).any() else None,
    "n_rare": int(rare.sum()), "n_common": int((~rare).sum()),
}
log(f"    c+ summary: {json.dumps(results['cplus_summary'])} spearman={results['cplus_spearman_trainfreq_vs_mse']}")

OUT.write_text(json.dumps(results, indent=1))
log(f"wrote {OUT}")
