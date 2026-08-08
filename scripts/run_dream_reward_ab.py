"""Controlled A/B/C: how the dream-training REWARD SOURCE affects real transfer.

All three dream agents are trained inside the SAME frozen world model, from the
same real dream-starts, with identical policy architecture, horizon, batch and
update budget. Only the reward/termination source differs:

  reward_head : trust the WM's learned reward + done heads (iteration-1 recipe;
                expected to farm the vanish/hallucination exploit -> dodging).
  ball_guard  : iteration-2 fix — zero reward + continuation in ball-less frames.
  geo         : iteration-3 lever — ignore the heads; read an honest reward +
                termination off decoded frame GEOMETRY (agents/geo_reward.py).

Each agent is then evaluated greedily on REAL Pong (unshaped points). The
headline is the transfer gap: dream-internal return vs real mean point, plus
paddle hits / episode (does the agent actually play, or dodge?).

CLI: python -m scripts.run_dream_reward_ab --wm checkpoints/wm_exp.pt \
        --updates 300 --batch 128 --eval-episodes 50
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import config
from agents.policy import load_policy
from agents.evaluate import eval_policy

ARMS = {
    "reward_head": [],
    "ball_guard": ["--ball-guard"],
    "geo": ["--geo-reward"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default=str(config.CKPT_DIR / "wm_exp.pt"))
    ap.add_argument("--data", default=str(config.DATA_DIR / "local"))
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--outdir", default=str(config.RESULTS_DIR / "exp_geo"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dev = config.get_device()

    summary = {"wm": args.wm, "updates": args.updates, "batch": args.batch,
               "eval_episodes": args.eval_episodes, "arms": {}}

    # ---- train each arm ----------------------------------------------------
    for arm, flags in ARMS.items():
        ckpt = config.CKPT_DIR / f"dream_{arm}.pt"
        print(f"\n===== TRAIN arm={arm} flags={flags} =====", flush=True)
        cmd = [sys.executable, "-m", "agents.train_dream",
               "--wm", args.wm, "--data", args.data,
               "--updates", str(args.updates), "--batch", str(args.batch),
               "--out", str(ckpt)] + flags
        subprocess.run(cmd, check=True)
        # the trainer wrote results/dream_train_log.json for THIS arm; snapshot it
        log = json.loads((config.RESULTS_DIR / "dream_train_log.json").read_text())
        (outdir / f"dream_log_{arm}.json").write_text(json.dumps(log))
        tail = log["dream_return"][-5:] if log["dream_return"] else [0.0]
        summary["arms"][arm] = {"dream_return_last5": float(np.mean(tail)),
                                "dream_len_last5": float(np.mean(log["dream_len"][-5:]))
                                if log["dream_len"] else 0.0,
                                "ckpt": str(ckpt)}

    # ---- evaluate each arm on REAL Pong ------------------------------------
    from evalutils import gifs
    for arm in ARMS:
        ckpt = config.CKPT_DIR / f"dream_{arm}.pt"
        print(f"\n===== EVAL arm={arm} on real Pong =====", flush=True)
        policy = load_policy(ckpt, dev).eval()
        stats, frames = eval_policy(policy, args.eval_episodes, dev, record_first=300)
        summary["arms"][arm].update(stats)
        if frames:
            gifs.save_gif(np.stack(frames), outdir / f"play_{arm}.gif", fps=30)
        di = summary["arms"][arm]["dream_return_last5"]
        print(f"[{arm}] real mean_point={stats['mean_point']:+.3f} "
              f"win={stats['win_rate']:.2f} hits/ep={stats['mean_hits']:.2f} "
              f"len={stats['mean_len']:.0f} | dream_return={di:+.3f} "
              f"gap={di - stats['mean_point']:+.3f}", flush=True)

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nAB_EXPERIMENT_OK", json.dumps(summary["arms"], indent=2))


if __name__ == "__main__":
    main()
