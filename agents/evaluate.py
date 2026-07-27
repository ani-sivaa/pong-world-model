"""Phase 3c — evaluate policies on REAL Pong; transfer comparison.

Reports per policy: mean points/episode (raw, unshaped), win/loss/truncation
counts, paddle hits, episode length. For the dream agent it also pulls the
dream-internal return from results/dream_train_log.json — the dream-vs-real gap
is the headline number. Flags suspected world-model exploitation when the
dream return is high but real performance is poor.

CLI: python -m agents.evaluate --scale full \
       --policies baseline=checkpoints/ppo_baseline.pt dream=checkpoints/dream_agent.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config
from agents.policy import load_policy
from pong.env import VecPong


@torch.no_grad()
def eval_policy(policy, n_episodes, dev, seed=123, record_first=0):
    """Greedy (argmax) evaluation with vectorized envs. Returns stats + frames."""
    n = min(32, n_episodes)
    env = VecPong(n, seed=seed)
    f = env.reset()
    stacks = np.repeat(f[:, None], config.FRAME_STACK, axis=1)
    done_eps, points, hits, lens = 0, [], [], []
    ep_hit = np.zeros(n); ep_len = np.zeros(n, dtype=np.int64)
    frames_rec = []
    while done_eps < n_episodes:
        x = torch.from_numpy(stacks).to(dev).float() / 255.0
        logits, _ = policy(x)
        a = logits.argmax(dim=1).cpu().numpy()
        f, r, d, info = env.step(a)
        if record_first:
            frames_rec.append(f[0].copy())
            if len(frames_rec) >= record_first:
                record_first = 0
        ep_hit += info["paddle_hit"]; ep_len += 1
        stacks = np.concatenate([stacks[:, 1:], f[:, None]], axis=1)
        if d.any():
            stacks[d] = f[d][:, None]
            for i in np.flatnonzero(d):
                points.append(int(info["point"][i]))
                hits.append(float(ep_hit[i])); lens.append(int(ep_len[i]))
                ep_hit[i] = ep_len[i] = 0
                done_eps += 1
    points, hits, lens = (np.array(points[:n_episodes]),
                          np.array(hits[:n_episodes]),
                          np.array(lens[:n_episodes]))
    return {
        "episodes": int(len(points)),
        "mean_point": float(points.mean()),
        "win_rate": float((points == 1).mean()),
        "loss_rate": float((points == -1).mean()),
        "trunc_rate": float((points == 0).mean()),
        "mean_hits": float(hits.mean()),
        "mean_len": float(lens.mean()),
    }, frames_rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="local", choices=list(config.SCALES))
    ap.add_argument("--policies", nargs="+", required=True,
                    help="name=ckpt_path pairs")
    ap.add_argument("--gif-steps", type=int, default=300)
    ap.add_argument("--out", default=None,
                    help="output JSON path (legacy default: results/transfer.json)")
    ap.add_argument("--tag", default=None,
                    help="artifact suffix; also selects dream_train_log_<tag>.json")
    ap.add_argument("--dream-log", default=None,
                    help="explicit dream training log used for the return gap")
    args = ap.parse_args()

    from evalutils import gifs

    config.seed_everything()
    dev = config.get_device()
    n_eps = config.SCALES[args.scale]["eval_episodes"]
    report = {}
    for spec in args.policies:
        name, path = spec.split("=", 1)
        policy = load_policy(path, dev).eval()
        stats, frames = eval_policy(policy, n_eps, dev,
                                    record_first=args.gif_steps)
        report[name] = stats
        if frames:
            suffix = f"_{args.tag}" if args.tag else ""
            gifs.save_gif(np.stack(frames),
                          config.RESULTS_DIR / f"agent_{name}{suffix}_play.gif", fps=30)
        print(f"[eval] {name}: {json.dumps(stats)}", flush=True)

    # dream-internal score for the headline gap
    dream_log = (Path(args.dream_log) if args.dream_log else
                 config.RESULTS_DIR / (
                     f"dream_train_log_{args.tag}.json" if args.tag
                     else "dream_train_log.json"))
    if "dream" in report and dream_log.exists():
        dl = json.loads(dream_log.read_text())
        tail = dl["dream_return"][-10:] if dl["dream_return"] else [0.0]
        report["dream_internal"] = {"mean_dream_return_last10": float(np.mean(tail))}
        gap_note = (
            "SUSPECTED WORLD-MODEL EXPLOITATION" if
            (np.mean(tail) > 0.5 and report["dream"]["mean_point"] < 0.0)
            else "no exploitation flag")
        report["dream_internal"]["exploit_flag"] = gap_note

    out = (Path(args.out) if args.out else config.RESULTS_DIR / (
        f"transfer_{args.tag}.json" if args.tag else "transfer.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("TRANSFER_EVAL_OK", json.dumps(report))


if __name__ == "__main__":
    main()
