"""Data collection: scripted eps-random play with VecPong -> data/<scale>/.

CLI: python -m pong.collect --scale smoke|local|full [--out DIR] [--if-missing]

Output (see INTERFACES.md "Data format"):
  frames.npy  uint8 [T, 64, 64]  frame BEFORE each action (frames[t] <-> actions[t])
  actions.npy uint8 [T]; rewards.npy float32 [T]; dones.npy bool [T]
  meta.json   {seed, T, n_episodes, env_constants, collect, ...}

Alignment with vectorized envs: transitions are stored as PER-ENV CONTIGUOUS
streams concatenated env-by-env, so within a stream frames[t+1] is the
post-action frame (or the auto-reset serve frame when dones[t]). The last
transition of every env's chunk gets dones=True FORCED (even if the episode
did not really end) so no consumer ever stitches a (t -> t+1) pair across two
envs' streams — see DECISIONS.env-builder.md.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import config
from pong.env import VecPong, scripted_action


def collect(scale: str, out_dir: Path, seed: int) -> dict:
    T = config.SCALES[scale]["transitions"]
    n = config.COLLECT["n_envs"]
    eps = config.COLLECT["eps_random"]
    E = config.ENV

    out_dir.mkdir(parents=True, exist_ok=True)
    env = VecPong(n, seed=seed)
    # policy RNG separate from the env's serve RNG (distinct stream, same root seed)
    pol_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])

    # per-env quotas: env i contributes quota[i] transitions at rows
    # [offset[i], offset[i] + quota[i]) of the concatenated stream.
    quota = np.full(n, T // n, np.int64)
    quota[: T % n] += 1
    offsets = np.concatenate([[0], np.cumsum(quota)[:-1]])
    steps = int(quota.max())

    # frames written straight into the final .npy via memmap (full = ~4.1 GB).
    frames_out = np.lib.format.open_memmap(
        out_dir / "frames.npy", mode="w+", dtype=np.uint8, shape=(T, E["H"], E["W"]))
    actions_out = np.zeros(T, np.uint8)
    rewards_out = np.zeros(T, np.float32)
    dones_out = np.zeros(T, bool)

    cur = env.reset()
    t0 = time.perf_counter()
    for s in range(steps):
        a = scripted_action(env.right_y, env.ball_y, eps, pol_rng)
        nxt, r, d, _ = env.step(a)
        active = s < quota                    # envs still under quota this step
        idx = (offsets + s)[active]
        frames_out[idx] = cur[active]
        actions_out[idx] = a[active].astype(np.uint8)
        rewards_out[idx] = r[active]
        dones_out[idx] = d[active]
        cur = nxt
    elapsed = time.perf_counter() - t0

    n_true_dones = int(dones_out.sum())
    # force done=True at every env-stream boundary (stream isolation, see docstring)
    dones_out[offsets + quota - 1] = True
    n_episodes = int(dones_out.sum())

    frames_out.flush()
    del frames_out
    np.save(out_dir / "actions.npy", actions_out)
    np.save(out_dir / "rewards.npy", rewards_out)
    np.save(out_dir / "dones.npy", dones_out)

    stats = dict(
        seed=seed, T=T, n_episodes=n_episodes,
        env_constants=dict(config.ENV), collect=dict(config.COLLECT),
        scale=scale, n_true_dones=n_true_dones,
        n_forced_boundary_dones=n_episodes - n_true_dones,
        mean_episode_len=T / n_episodes,
        reward_sum=float(rewards_out.sum()),
        points_plus=int((rewards_out >= config.ENV["r_score"]).sum()),
        points_minus=int((rewards_out <= config.ENV["r_concede"]).sum()),
        elapsed_sec=elapsed, transitions_per_sec=T / elapsed,
    )
    (out_dir / "meta.json").write_text(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", required=True, choices=list(config.SCALES))
    ap.add_argument("--out", default=None, help="output dir (default data/<scale>)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--if-missing", action="store_true",
                    help="skip if meta.json exists with matching T")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else config.DATA_DIR / args.scale
    meta_path = out_dir / "meta.json"
    if args.if_missing and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("T") == config.SCALES[args.scale]["transitions"]:
            print(f"[collect] {out_dir} already has T={meta['T']}; skipping.")
            return
        print(f"[collect] {out_dir} exists but T mismatch; re-collecting.")

    stats = collect(args.scale, out_dir, args.seed)
    print(f"[collect] scale={args.scale} -> {out_dir}")
    print(f"  T={stats['T']:,}  episodes={stats['n_episodes']} "
          f"(true dones={stats['n_true_dones']}, forced boundary="
          f"{stats['n_forced_boundary_dones']})  mean_ep_len={stats['mean_episode_len']:.1f}")
    print(f"  reward_sum={stats['reward_sum']:.1f}  +points={stats['points_plus']} "
          f"-points={stats['points_minus']}")
    print(f"  {stats['elapsed_sec']:.1f}s  ->  "
          f"{stats['transitions_per_sec']:,.0f} transitions/sec")


if __name__ == "__main__":
    main()
