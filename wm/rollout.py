"""Autoregressive rollout ("dreaming") + drift evaluation.

The key experiment: feed the model's own predictions back in and measure how
long the dream stays coherent with no real game underneath.

CLI:
  python -m wm.rollout --ckpt checkpoints/wm_v1.pt --scale local \
      [--data DIR] [--tag main] [--gif-lengths 15 30 60]
Outputs: results/drift_curve_<tag>.png, results/drift_<tag>.json,
         results/rollouts/<tag>_h<H>_sample<i>.gif
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config
from wm.data import TransitionData
from wm.model import load_wm


@torch.no_grad()
def dream(wm, stack0: torch.Tensor, horizon: int, actions_fn):
    """Roll the world model forward on its own predictions.

    stack0: float [B,4,64,64] in [0,1]. actions_fn(stack, j) -> long [B].
    Returns (frames [B,H,64,64] float probs, rewards [B,H], done_probs [B,H]);
    reward/done tensors are zeros for a v1 model (no heads).
    """
    B, H = stack0.shape[0], horizon
    dev = stack0.device
    stack = stack0.clone()
    frames = torch.zeros(B, H, 64, 64, device=dev)
    rewards = torch.zeros(B, H, device=dev)
    done_probs = torch.zeros(B, H, device=dev)
    actions = torch.zeros(B, H, dtype=torch.long, device=dev)
    for j in range(H):
        a = actions_fn(stack, j)
        logits, r, d = wm(stack, a)
        probs = torch.sigmoid(logits[:, 0])            # [B,64,64]
        frames[:, j] = probs
        actions[:, j] = a
        if r is not None:
            rewards[:, j] = r
            done_probs[:, j] = torch.sigmoid(d)
        stack = torch.cat([stack[:, 1:], probs[:, None]], dim=1)
    return frames, rewards, done_probs, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.CKPT_DIR / "wm_v1.pt"))
    ap.add_argument("--scale", default="local", choices=list(config.SCALES))
    ap.add_argument("--data", default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--gif-lengths", type=int, nargs="*", default=[15, 30, 60])
    ap.add_argument("--gif-samples", type=int, default=3)
    ap.add_argument("--with-heads", action="store_true")
    args = ap.parse_args()

    from evalutils import gifs, metrics, plots  # heavier import kept local

    config.seed_everything()
    dev = config.get_device()
    sc = config.SCALES[args.scale]
    horizon = sc["rollout_eval_h"]
    data_dir = args.data or (config.DATA_DIR / args.scale)

    wm = load_wm(args.ckpt, dev, with_heads=args.with_heads).eval()
    data = TransitionData(data_dir)
    stacks, acts, real = data.val_windows(args.n_windows, horizon)

    stack0 = torch.from_numpy(stacks.astype(np.float32) / 255.0).to(dev)
    acts_t = torch.from_numpy(acts).to(dev)
    dreamed, _, _, _ = dream(wm, stack0, horizon,
                             lambda s, j: acts_t[:, j])   # replay LOGGED actions

    # ---- drift curve: per-step MSE averaged over windows -----------------
    dreamed_u8 = (dreamed.cpu().numpy() * 255).astype(np.uint8)
    curves = np.stack([metrics.mse_curve(real[i], dreamed_u8[i])
                       for i in range(len(real))])       # [N,H]
    mean_curve = curves.mean(axis=0)

    out_json = config.RESULTS_DIR / f"drift_{args.tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "h": list(range(1, horizon + 1)),
        "mse_mean": mean_curve.tolist(),
        "mse_std": curves.std(axis=0).tolist(),
        "n_windows": int(len(real)), "ckpt": args.ckpt,
    }))
    plots.line_plot(
        list(range(1, horizon + 1)), {"mean MSE": mean_curve},
        title=f"Dream drift — prediction error vs rollout step ({args.tag})",
        xlabel="rollout step (autoregressive)", ylabel="per-pixel MSE",
        out_path=config.RESULTS_DIR / f"drift_curve_{args.tag}.png", log_y=True)

    # ---- real vs dreamed side-by-side GIFs -------------------------------
    for h in args.gif_lengths:
        if h > horizon:
            continue
        for i in range(min(args.gif_samples, len(real))):
            diff = np.abs(real[i][:h].astype(np.int16)
                          - dreamed_u8[i][:h].astype(np.int16)).astype(np.uint8)
            gifs.side_by_side(
                [real[i][:h], dreamed_u8[i][:h], diff],
                ["real", "dream", "|diff|"],
                config.RESULTS_DIR / "rollouts" / f"{args.tag}_h{h}_sample{i}.gif",
                fps=15)
    print(f"ROLLOUT_EVAL_OK tag={args.tag} "
          f"mse@1={mean_curve[0]:.5f} mse@{horizon}={mean_curve[-1]:.5f}")


if __name__ == "__main__":
    main()
