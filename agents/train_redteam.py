"""Train a policy in imagination to expose world-model ensemble failures.

The frozen ensemble supplies only an intrinsic red-team score: frame, reward,
and done disagreement plus disappearance of a previously visible ball. No Pong
reward is optimized. Policy gradients use REINFORCE, so gradients never pass
through a world model.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from agents.policy import PolicyNet
from wm.data import TransitionData
from wm.model import load_wm_ensemble


def ball_present(frames: torch.Tensor) -> torch.Tensor:
    """Return a hard ball-presence indicator for frames shaped [...,1,64,64]."""
    pixels = (frames[..., 0, :, :] > 0.5).float().clone()
    pixels[..., 0, :] = 0
    pixels[..., 63, :] = 0
    pixels[..., :, 2:4] = 0
    pixels[..., :, 60:62] = 0
    return (pixels.sum(dim=(-2, -1)) >= 2).float()


def redteam_score(prediction, previous_frame: torch.Tensor,
                  reward_coef: float, frame_coef: float, done_coef: float,
                  disappearance_coef: float):
    """Intrinsic score and components, deliberately independent of mean reward."""
    previous_alive = ball_present(previous_frame)
    member_alive = ball_present(prediction.member_frames)
    disappearance = previous_alive * (1.0 - member_alive.mean(0))
    components = {
        "reward_disagreement": prediction.reward_std,
        "frame_disagreement": prediction.frame_mse,
        "done_disagreement": prediction.done_disagreement,
        "ball_disappearance": disappearance,
    }
    score = (
        reward_coef * components["reward_disagreement"]
        + frame_coef * components["frame_disagreement"]
        + done_coef * components["done_disagreement"]
        + disappearance_coef * components["ball_disappearance"]
    )
    return score, components


def _flatten(groups):
    return [item for group in (groups or []) for item in group]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", default="local", choices=list(config.SCALES))
    ap.add_argument("--wm", nargs="+", action="append", required=True,
                    help="headed WM checkpoints; repeat or pass several paths")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=str(config.CKPT_DIR / "redteam_agent.pt"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--frame-mode", choices=("mean", "sample"),
                    default=config.REDTEAM["frame_mode"])
    ap.add_argument("--latent-mode", choices=("mean", "sample"), default="mean",
                    help="use prior mean or sample stochastic WM dynamics latents")
    ap.add_argument("--reward-disagreement-coef", type=float,
                    default=config.REDTEAM["reward_disagreement_coef"])
    ap.add_argument("--frame-disagreement-coef", type=float,
                    default=config.REDTEAM["frame_disagreement_coef"])
    ap.add_argument("--done-disagreement-coef", type=float,
                    default=config.REDTEAM["done_disagreement_coef"])
    ap.add_argument("--ball-disappearance-coef", type=float,
                    default=config.REDTEAM["ball_disappearance_coef"])
    ap.add_argument("--entropy-coef", type=float, default=config.REDTEAM["ent_coef"])
    ap.add_argument("--max-seconds", type=int, default=config.CAPS["dream"])
    args = ap.parse_args()

    config.seed_everything(args.seed)
    device = config.get_device()
    cfg = config.REDTEAM
    updates = args.updates or config.SCALES[args.scale]["dream_updates"]
    batch = args.batch or cfg["batch"]
    horizon = args.horizon or cfg["horizon"]
    wm_paths = _flatten(args.wm)
    data_dir = args.data or config.DATA_DIR / args.scale

    for path in wm_paths:
        state = torch.load(path, map_location="cpu", weights_only=True)["model"]
        if not (any(key.startswith("reward_head.") for key in state)
                and any(key.startswith("done_head.") for key in state)):
            raise ValueError(f"red-team training requires headed WM: {path}")
    ensemble = load_wm_ensemble(wm_paths, device, with_heads=True).eval()
    for parameter in ensemble.parameters():
        parameter.requires_grad_(False)
    policy = PolicyNet().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"], eps=1e-5)
    data = TransitionData(data_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 2_000_003)
    log = {name: [] for name in (
        "update", "intrinsic_return", "entropy", "loss_pi", "loss_v",
        "reward_disagreement", "frame_disagreement", "done_disagreement",
        "ball_disappearance", "aleatoric_frame_mse")}
    started = time.time()
    last_update = 0

    print(f"[redteam] device={device.type} members={len(wm_paths)} "
          f"updates={updates} batch={batch} horizon={horizon}", flush=True)
    for update in range(1, updates + 1):
        last_update = update
        starts, *_ = data.sample(batch, 1)
        stack = torch.from_numpy(starts.astype(np.float32) / 255.0).to(device)
        logps, values, entropies, scores = [], [], [], []
        component_steps = {name: [] for name in (
            "reward_disagreement", "frame_disagreement", "done_disagreement",
            "ball_disappearance", "aleatoric_frame_mse")}
        for _ in range(horizon):
            logits, value = policy(stack)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            logps.append(distribution.log_prob(action))
            values.append(value)
            entropies.append(distribution.entropy())
            with torch.no_grad():
                previous_frame = stack[:, -1:]
                stack, prediction = ensemble.rollout_step(
                    stack, action, frame_mode=args.frame_mode,
                    latent_mode=args.latent_mode, generator=generator)
                score, components = redteam_score(
                    prediction, previous_frame,
                    args.reward_disagreement_coef,
                    args.frame_disagreement_coef,
                    args.done_disagreement_coef,
                    args.ball_disappearance_coef)
                components["aleatoric_frame_mse"] = prediction.aleatoric_frame_mse
            scores.append(score)
            for name, values_for_name in component_steps.items():
                values_for_name.append(components[name])

        returns = [None] * horizon
        running = torch.zeros(batch, device=device)
        for step in reversed(range(horizon)):
            running = scores[step] + cfg["gamma"] * running
            returns[step] = running
        returns_t = torch.stack(returns)
        values_t = torch.stack(values)
        logps_t = torch.stack(logps)
        entropy_t = torch.stack(entropies)
        advantage = (returns_t - values_t).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        loss_pi = -(logps_t * advantage).mean()
        loss_v = F.mse_loss(values_t, returns_t.detach())
        loss = loss_pi + cfg["vf_coef"] * loss_v - args.entropy_coef * entropy_t.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"])
        optimizer.step()

        if update == 1 or update % 10 == 0:
            log["update"].append(update)
            log["intrinsic_return"].append(float(torch.stack(scores).sum(0).mean()))
            log["entropy"].append(float(entropy_t.mean()))
            log["loss_pi"].append(float(loss_pi))
            log["loss_v"].append(float(loss_v))
            for name, values_for_name in component_steps.items():
                log[name].append(float(torch.stack(values_for_name).mean()))
            print(f"  update {update}/{updates} intrinsic="
                  f"{log['intrinsic_return'][-1]:.4f} "
                  f"disappear={log['ball_disappearance'][-1]:.4f}", flush=True)
        if time.time() - started > args.max_seconds:
            print(f"[redteam] wall-clock cap at update {update}", flush=True)
            break

    checkpoint = {
        "model": {key: value.cpu() for key, value in policy.state_dict().items()},
        "step": last_update,
        "config": dict(cfg, wm=wm_paths, seed=args.seed,
                       reward_disagreement_coef=args.reward_disagreement_coef,
                       frame_disagreement_coef=args.frame_disagreement_coef,
                       done_disagreement_coef=args.done_disagreement_coef,
                       ball_disappearance_coef=args.ball_disappearance_coef,
                       frame_mode=args.frame_mode, latent_mode=args.latent_mode,
                       entropy_coef=args.entropy_coef),
    }
    torch.save(checkpoint, out)
    suffix = f"_{args.tag}" if args.tag else ""
    result_path = config.RESULTS_DIR / f"redteam_train_log{suffix}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(log))
    print(f"REDTEAM_OK updates={last_update} ckpt={out} log={result_path}")


if __name__ == "__main__":
    main()
