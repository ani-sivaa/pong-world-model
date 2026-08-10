"""Validate the geometry-grounded reward oracle against the TRUE env signal.

The geo oracle (agents/geo_reward.py) reads reward + termination off decoded
frame geometry so dream training need not trust the WM's miscalibrated
reward/done heads. Before we trust it inside imperfect dreams, it must agree
with the real env on REAL transitions (where ground-truth reward/done is known).

Events are plane crossings, so we score PER RALLY (not per step, which would
be sensitive to 1-frame quantization offsets between "ball left the playfield"
and "ball fully off-screen"). For every real episode we ask:
  * terminal event: does geo end the rally with the correct sign
    (-1 concede / +1 score / 0 truncation) as the env's true point?
  * hit shaping: how does geo's summed +r_hit compare to the env's true count
    of agent paddle hits (should be close; this is dense shaping, not terminal)?
  * honesty: does geo ever pay reward in a genuinely ball-less frame?
"""
import numpy as np
import torch

import config
from pong.env import VecPong, scripted_action
from agents.geo_reward import geo_signal, _ball_stats

_E = config.ENV


def main():
    config.seed_everything()
    n, steps = 64, 1500
    env = VecPong(n, seed=7)
    rng = np.random.default_rng(7)
    f = env.reset()
    # rolling 3-frame window (prev, cur); nxt is produced by each step
    prev = f.copy()
    cur = f.copy()

    # per-env accumulators for the CURRENT rally
    geo_hits = np.zeros(n)
    env_hits = np.zeros(n)
    geo_term_sign = np.zeros(n)        # last geo terminal reward sign this rally
    ballless_pay = 0                    # geo reward emitted in a ball-less cur frame

    # per-rally records
    rec_env_point, rec_geo_sign = [], []
    rec_geo_hits, rec_env_hits = [], []

    for _ in range(steps):
        a = scripted_action(env.right_y, env.ball_y, 0.4, rng)
        nxt_frame, r, d, info = env.step(a)
        nxt = nxt_frame.copy()
        if d.any():                     # use the true terminal frame for geometry
            nxt[d] = info["terminal_frame"][d]

        pt = torch.from_numpy(prev[:, None].astype(np.float32) / 255.0)
        ct = torch.from_numpy(cur[:, None].astype(np.float32) / 255.0)
        nt = torch.from_numpy(nxt[:, None].astype(np.float32) / 255.0)
        g_rew, g_cont = geo_signal(pt, ct, nt)
        g_rew = g_rew.numpy()

        cur_present, _, _ = _ball_stats(ct)
        ballless_pay += int(((g_rew != 0) & (~cur_present.numpy())).sum())

        geo_hits += (np.isclose(g_rew, _E["r_hit"])).astype(float)
        env_hits += info["paddle_hit"].astype(float)
        term = np.isclose(g_rew, _E["r_concede"]) | np.isclose(g_rew, _E["r_score"])
        geo_term_sign = np.where(term, np.sign(g_rew), geo_term_sign)

        # advance rolling window
        prev = cur
        cur = nxt_frame.copy()

        for i in np.flatnonzero(d):
            rec_env_point.append(int(info["point"][i]))
            rec_geo_sign.append(int(geo_term_sign[i]))
            rec_geo_hits.append(float(geo_hits[i]))
            rec_env_hits.append(float(env_hits[i]))
            geo_hits[i] = env_hits[i] = geo_term_sign[i] = 0
            prev[i] = cur[i] = nxt_frame[i]     # reset window on the fresh serve

    ep = np.array(rec_env_point)
    gs = np.array(rec_geo_sign)
    n_ep = len(ep)
    concede = ep == -1
    score = ep == 1
    trunc = ep == 0

    def acc(mask, label):
        if mask.sum() == 0:
            print(f"  {label:9s} n=0"); return
        correct = (gs[mask] == np.sign(ep[mask]) if label != "trunc"
                   else gs[mask] == 0)
        print(f"  {label:9s} n={int(mask.sum()):4d}  terminal-sign accuracy="
              f"{float(np.mean(correct)):.3f}")

    print(f"GEO-REWARD PER-RALLY VALIDATION: {n_ep} rallies "
          f"({steps} steps x {n} envs)")
    acc(concede, "concede")
    acc(score, "score")
    acc(trunc, "trunc")
    gh, eh = np.array(rec_geo_hits), np.array(rec_env_hits)
    print(f"  hits/rally: geo={gh.mean():.2f} env={eh.mean():.2f} "
          f"(ratio {gh.sum()/max(eh.sum(),1):.2f})")
    print(f"  ball-less frames paid reward: {ballless_pay} (must be 0)")

    concede_acc = float(np.mean(gs[concede] == -1)) if concede.sum() else 1.0
    ok = concede_acc >= 0.9 and ballless_pay == 0
    print("GEO_VALIDATE_OK" if ok else "GEO_VALIDATE_WEAK")


if __name__ == "__main__":
    main()
