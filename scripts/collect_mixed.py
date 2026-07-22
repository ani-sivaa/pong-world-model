"""Iteration-2 data collection: mixed behavior policies (the Dreamer/DAgger move).

The v1 dataset was collected by one scripted tracker (+20% random), so the
world model only learned physics along tracker-style trajectories — and the
dream policy exploited it off-distribution. This collector mixes:

  tracker   40%  — the original scripted policy (keeps old coverage)
  baseline  35%  — the trained PPO agent, SAMPLING from its logits (competent,
                   diverse right-paddle play — the states a good policy visits)
  random    15%  — uniform actions (broad coverage)
  dream     10%  — the current dream agent (its own visited states; ensures the
                   retrained WM is accurate exactly where this policy goes)

Each segment is collected with per-env contiguous streams and a forced
done=True at every env-chunk AND segment boundary (same convention as
pong/collect.py), then all segments are concatenated. Output format is
contract-identical (INTERFACES.md).

CLI: python -m scripts.collect_mixed --scale full --out data/full_v2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config
from agents.policy import load_policy
from pong.env import VecPong, scripted_action

MIX = [("tracker", 0.40), ("baseline", 0.35), ("random", 0.15), ("dream", 0.10)]


def collect_segment(kind, n_transitions, seed, dev):
    n = config.COLLECT["n_envs"]
    env = VecPong(n, seed=seed)
    rng = np.random.default_rng(seed)
    f = env.reset()
    stacks = np.repeat(f[:, None], config.FRAME_STACK, axis=1)

    policy = None
    if kind in ("baseline", "dream"):
        ckpt = "ppo_baseline.pt" if kind == "baseline" else "dream_agent.pt"
        policy = load_policy(config.CKPT_DIR / ckpt, dev).eval()

    steps = n_transitions // n
    frames = np.zeros((n, steps, 64, 64), np.uint8)
    actions = np.zeros((n, steps), np.uint8)
    rewards = np.zeros((n, steps), np.float32)
    dones = np.zeros((n, steps), bool)

    cur = f
    for t in range(steps):
        if kind == "tracker":
            a = scripted_action(env.right_y, env.ball_y,
                                config.COLLECT["eps_random"], rng)
        elif kind == "random":
            a = rng.integers(0, config.N_ACTIONS, size=n)
        else:  # policy-driven, SAMPLED for diversity
            with torch.no_grad():
                x = torch.from_numpy(stacks).to(dev).float() / 255.0
                logits, _ = policy(x)
                a = torch.distributions.Categorical(logits=logits).sample()
                a = a.cpu().numpy()
        frames[:, t] = cur
        actions[:, t] = a
        nxt, r, d, info = env.step(np.asarray(a))
        rewards[:, t] = r
        dones[:, t] = d
        stacks = np.concatenate([stacks[:, 1:], nxt[:, None]], axis=1)
        if d.any():
            stacks[d] = nxt[d][:, None]
        cur = nxt

    dones[:, -1] = True  # forced env-chunk boundary done (contract convention)
    return (frames.reshape(-1, 64, 64), actions.reshape(-1),
            rewards.reshape(-1), dones.reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full", choices=list(config.SCALES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config.seed_everything()
    dev = config.get_device()
    total = config.SCALES[args.scale]["transitions"]
    out = Path(args.out or (config.DATA_DIR / f"{args.scale}_v2"))
    out.mkdir(parents=True, exist_ok=True)

    parts = {k: [] for k in ("frames", "actions", "rewards", "dones")}
    seg_stats = {}
    for i, (kind, frac) in enumerate(MIX):
        n_seg = int(total * frac)
        f, a, r, d = collect_segment(kind, n_seg, seed=config.SEED + 100 + i, dev=dev)
        parts["frames"].append(f); parts["actions"].append(a)
        parts["rewards"].append(r); parts["dones"].append(d)
        seg_stats[kind] = {"T": int(len(a)), "reward_sum": float(r.sum()),
                           "true_dones": int(d.sum() - config.COLLECT["n_envs"])}
        print(f"[collect_mixed] {kind}: T={len(a)} reward_sum={r.sum():.1f}",
              flush=True)

    arrs = {k: np.concatenate(v) for k, v in parts.items()}
    np.save(out / "frames.npy", arrs["frames"])
    np.save(out / "actions.npy", arrs["actions"])
    np.save(out / "rewards.npy", arrs["rewards"].astype(np.float32))
    np.save(out / "dones.npy", arrs["dones"])
    T = len(arrs["actions"])
    (out / "meta.json").write_text(json.dumps({
        "seed": config.SEED, "T": T,
        "n_episodes": int(arrs["dones"].sum()),
        "mix": {k: f for k, f in MIX}, "segments": seg_stats,
        "env_constants": config.ENV, "collect": config.COLLECT,
    }, indent=2))
    print(f"COLLECT_MIXED_OK T={T} -> {out}")


if __name__ == "__main__":
    main()
