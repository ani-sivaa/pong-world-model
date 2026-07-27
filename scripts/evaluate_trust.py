"""Evaluate whether headed world models are trustworthy enough for dreaming.

The report intentionally separates measurements from gates. Missing event
classes produce ``available: false`` metrics rather than invented calibration
numbers. By default a failing report is still written and exits successfully;
use ``--enforce-trust`` to make the overall gate control the process exit code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config
from agents.policy import load_policy
from pong.env import VecPong, scripted_action
from wm.data import EVENT_NAMES, TransitionData
from wm.model import StochasticWorldModel, load_wm_ensemble


def _metric(value=None, *, reason=None, **extra):
    available = value is not None and np.isfinite(value)
    result = {"available": bool(available)}
    if available:
        result["value"] = float(value)
    else:
        result["reason"] = reason or "metric is not defined for these samples"
    result.update(extra)
    return result


def _rank(values):
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def rank_correlation(x, y):
    """Tie-aware Spearman correlation, or None for constant/short inputs."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def binary_auroc(scores, labels):
    """Rank-based AUROC, or None when either class is absent."""
    scores = np.asarray(scores)
    labels = np.asarray(labels, bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rank(scores) + 1.0
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def aggregate_gates(metrics, thresholds):
    """Apply explicit max/min thresholds and return individual + overall gates."""
    gates = {}
    for name, threshold in thresholds.items():
        metric_name, direction = name.rsplit("_", 1)
        measurement = metrics.get(metric_name, _metric(reason="measurement missing"))
        if not measurement.get("available", False):
            gates[metric_name] = {
                "available": False, "pass": None, "threshold": float(threshold),
                "reason": measurement.get("reason", "measurement unavailable"),
            }
            continue
        value = float(measurement["value"])
        passed = value <= threshold if direction == "max" else value >= threshold
        gates[metric_name] = {
            "available": True, "pass": bool(passed), "value": value,
            "operator": "<=" if direction == "max" else ">=",
            "threshold": float(threshold),
        }
    evaluated = [gate["pass"] for gate in gates.values() if gate["available"]]
    return gates, bool(evaluated) and all(evaluated)


def _tensor_stacks(stacks, device):
    return torch.from_numpy(np.asarray(stacks)).to(device).float() / 255.0


@torch.no_grad()
def _one_step(ensemble, stacks, actions, device):
    x = _tensor_stacks(stacks, device)
    a = torch.from_numpy(np.asarray(actions)).to(device, torch.long)
    prediction = ensemble.predict(x, a)
    return {
        "frame": prediction.mean_frame[:, 0].cpu().numpy(),
        "reward": prediction.mean_reward.cpu().numpy(),
        "done": prediction.mean_done_prob.cpu().numpy(),
        "uncertainty": (
            prediction.frame_mse + prediction.reward_std
            + prediction.done_disagreement).cpu().numpy(),
    }


def _class_metric(values, mask, label):
    count = int(np.asarray(mask).sum())
    if count == 0:
        return _metric(reason=f"no {label} samples", count=0)
    return _metric(np.asarray(values)[mask].mean(), count=count)


def calibration_metrics(prediction, targets, events):
    real_frame, real_reward, real_done = targets
    frame_error = ((prediction["frame"] - real_frame) ** 2).mean((1, 2))
    reward_error = np.abs(prediction["reward"] - real_reward)
    done_error = (prediction["done"] - real_done) ** 2
    total_error = frame_error + reward_error + np.abs(prediction["done"] - real_done)
    nonzero_reward = real_reward != 0
    done_mask = real_done.astype(bool)

    result = {
        "frame_mse": _metric(frame_error.mean()),
        "reward_mae": _metric(reward_error.mean()),
        "done_brier": _metric(done_error.mean()),
        "reward_nonzero_mae": _class_metric(
            reward_error, nonzero_reward, "nonzero-reward"),
        "reward_zero_mae": _class_metric(
            reward_error, ~nonzero_reward, "zero-reward"),
        "done_positive_brier": _class_metric(done_error, done_mask, "done-positive"),
        "done_negative_brier": _class_metric(done_error, ~done_mask, "done-negative"),
    }
    for code, name in EVENT_NAMES.items():
        mask = events == code
        result[f"event_{name}_reward_mae"] = _class_metric(
            reward_error, mask, f"event '{name}'")

    corr = rank_correlation(prediction["uncertainty"], total_error)
    result["uncertainty_rank_corr"] = _metric(
        corr, reason="uncertainty or real error is constant")
    high_error = total_error > np.median(total_error)
    auroc = binary_auroc(prediction["uncertainty"], high_error)
    result["uncertainty_high_error_auroc"] = _metric(
        auroc, reason="high-error split or uncertainty ranking is unavailable")
    result["_arrays"] = {
        "frame_error": frame_error, "reward_error": reward_error,
        "done_error": done_error,
    }
    return result


def _has_ball(frames):
    """Conservative visible-ball detector excluding walls and paddle columns."""
    pixels = np.asarray(frames) > 0.5
    pixels[:, (0, config.ENV["H"] - 1), :] = False
    for x in (config.ENV["left_x"], config.ENV["right_x"]):
        pixels[:, :, x:x + config.ENV["paddle_w"]] = False
    return pixels.any(axis=(1, 2))


def invariant_metrics(predicted_frames, predicted_rewards):
    has_ball = _has_ball(predicted_frames)
    # The smallest true positive reward is the 0.1 paddle-hit shaping reward.
    positive = np.asarray(predicted_rewards) > 0.05
    count = int((~has_ball).sum())
    rate = float((positive & ~has_ball).sum() / count) if count else None
    wall_intensity = np.concatenate([
        predicted_frames[:, 0, :].reshape(-1),
        predicted_frames[:, -1, :].reshape(-1),
    ]).mean()
    return {
        "ballless_positive_rate": _metric(
            rate, reason="all sampled predicted frames contain a visible ball",
            ballless_frames=count,
            violations=int((positive & ~has_ball).sum())),
        "frame_probability_bounds": _metric(float(
            ((predicted_frames >= 0) & (predicted_frames <= 1)).mean())),
        "mean_wall_intensity": _metric(wall_intensity),
    }


@torch.no_grad()
def natural_rollout(ensemble, data, horizon, count, device):
    used_horizon = int(horizon)
    while used_horizon > 0:
        try:
            stacks, actions, real = data.val_windows(count, used_horizon)
            break
        except ValueError:
            used_horizon //= 2
    if used_horizon == 0:
        return _metric(reason="no done-free validation rollout windows"), [], []
    dream = _tensor_stacks(stacks, device)
    errors, rewards, frames = [], [], []
    for step in range(used_horizon):
        a = torch.from_numpy(actions[:, step]).to(device, torch.long)
        dream, prediction = ensemble.rollout_step(
            dream, a, frame_mode="mean", latent_mode="mean")
        frame = dream[:, -1].cpu().numpy()
        errors.append(float(((frame - real[:, step] / 255.0) ** 2).mean()))
        rewards.extend(prediction.rollout_reward.cpu().tolist())
        frames.extend(frame)
    return _metric(np.mean(errors), horizon=used_horizon,
                   curve=errors, windows=len(stacks)), frames, rewards


@torch.no_grad()
def policy_lockstep(ensemble, policy, horizon, seed, device):
    n = 8
    env = VecPong(n, seed)
    current = env.reset()
    rng = np.random.default_rng(seed)
    history = [current]
    for _ in range(config.FRAME_STACK - 1):
        warm_action = scripted_action(env.right_y, env.ball_y, 0.0, rng)
        current, _, _, _ = env.step(warm_action)
        history.append(current)
    real_stack = np.stack(history, axis=1)
    dream_stack = _tensor_stacks(real_stack, device)
    errors, reward_gaps, frames, rewards = [], [], [], []
    for _ in range(horizon):
        if policy is None:
            actions = scripted_action(env.right_y, env.ball_y, 0.0, rng)
        else:
            logits, _ = policy(_tensor_stacks(real_stack, device))
            actions = logits.argmax(1).cpu().numpy()
        nxt, real_reward, done, _ = env.step(actions)
        action_t = torch.from_numpy(actions).to(device, torch.long)
        dream_stack, prediction = ensemble.rollout_step(
            dream_stack, action_t, frame_mode="mean", latent_mode="mean")
        dream_frame = dream_stack[:, -1].cpu().numpy()
        errors.extend(((dream_frame - nxt / 255.0) ** 2).mean((1, 2)).tolist())
        reward_gaps.extend(np.abs(
            prediction.rollout_reward.cpu().numpy() - real_reward).tolist())
        frames.extend(dream_frame)
        rewards.extend(prediction.rollout_reward.cpu().tolist())
        real_stack = np.concatenate([real_stack[:, 1:], nxt[:, None]], axis=1)
        if done.any():
            real_stack[done] = nxt[done, None]
            dream_stack[torch.from_numpy(done).to(device)] = _tensor_stacks(
                real_stack[done], device)
    return {
        "mse": _metric(np.mean(errors), steps=horizon, transitions=len(errors)),
        "reward_gap": _metric(np.mean(reward_gaps), count=len(reward_gaps)),
        "frames": frames, "rewards": rewards,
    }


@torch.no_grad()
def forced_scenarios(ensemble, device):
    reports, all_errors, frames, rewards = {}, [], [], []
    setups = {
        "intercept": {"right_y": 26.0, "ball_y": 28.0},
        "miss": {"right_y": 1.0, "ball_y": 50.0},
    }
    for index, (name, setup) in enumerate(setups.items()):
        env = VecPong(1, 900 + index)
        env.ball_x[:] = 57.0
        env.ball_vx[:] = config.ENV["ball_vx"]
        env.ball_y[:] = setup["ball_y"]
        env.ball_vy[:] = 0.0
        env.right_y[:] = setup["right_y"]
        # Build a physically consistent history ending at x=57 so velocity is
        # visible to the model instead of repeating one static frame.
        env.ball_x[:] = 52.5
        history = [env._render()]
        for _ in range(config.FRAME_STACK - 1):
            current, _, _, _ = env.step(np.array([0]))
            history.append(current)
        real_stack = np.stack(history, axis=1)
        dream_stack = _tensor_stacks(real_stack, device)
        errors, real_rewards, predicted_rewards, event = [], [], [], None
        for _ in range(6):
            nxt, reward, done, info = env.step(np.array([0]))
            dream_stack, prediction = ensemble.rollout_step(
                dream_stack, torch.zeros(1, dtype=torch.long, device=device),
                frame_mode="mean", latent_mode="mean")
            dream_frame = dream_stack[:, -1].cpu().numpy()
            errors.append(float(((dream_frame - nxt / 255.0) ** 2).mean()))
            real_rewards.append(float(reward[0]))
            predicted_rewards.append(float(prediction.rollout_reward[0].cpu()))
            frames.append(dream_frame[0])
            rewards.append(predicted_rewards[-1])
            if info["paddle_hit"][0]:
                event = "hit"
            if info["point"][0]:
                event = "concede" if info["point"][0] < 0 else "score"
            if done[0] or event == "hit":
                break
        all_errors.extend(errors)
        reports[name] = {
            "mse": _metric(np.mean(errors), steps=len(errors)),
            "observed_real_event": event,
            "real_rewards": real_rewards,
            "predicted_rewards": predicted_rewards,
        }
    return _metric(np.mean(all_errors)), reports, frames, rewards


@torch.no_grad()
def stochastic_diagnostics(ensemble, stacks, actions, samples, seed, device):
    if samples <= 0 or not any(
            isinstance(member, StochasticWorldModel) for member in ensemble.members):
        return {"available": False, "reason": (
            "sampling disabled" if samples <= 0 else "ensemble is deterministic")}
    x = _tensor_stacks(stacks, device)
    a = torch.from_numpy(np.asarray(actions)).to(device, torch.long)
    values = []
    for sample in range(samples):
        _, prediction = ensemble.rollout_step(
            x, a, latent_mode="sample",
            generator=torch.Generator().manual_seed(seed + sample))
        values.append(float(prediction.aleatoric_frame_mse.mean().cpu()))
    return {"available": True, "samples": samples,
            "mean_prior_sample_frame_mse": float(np.mean(values))}


def _parse_thresholds(scale, overrides):
    thresholds = dict(config.TRUST[scale])
    for item in overrides:
        name, value = item.split("=", 1)
        if name not in thresholds:
            raise ValueError(f"unknown trust threshold {name!r}")
        thresholds[name] = float(value)
    return thresholds


def evaluate(args):
    config.seed_everything(args.seed)
    device = config.get_device()
    data = TransitionData(args.data, seed=args.seed)
    ensemble = load_wm_ensemble(args.wm, device, with_heads=True).eval()
    policy = load_policy(args.policy, device).eval() if args.policy else None
    sample_count = args.samples or {
        "smoke": 128, "local": 512, "full": 2048}[args.scale]

    natural = data.sample(
        sample_count, 1, val=True, sampler="natural", return_info=True)
    event = data.sample(
        sample_count, 1, val=False, sampler="balanced", return_info=True)
    batches, infos = zip(natural, event)
    stacks = np.concatenate([item[0] for item in batches])
    actions = np.concatenate([item[1][:, 0] for item in batches])
    real_frames = np.concatenate([item[2][:, 0] for item in batches]
                                 ).astype(np.float32) / 255.0
    real_rewards = np.concatenate([item[3][:, 0] for item in batches])
    real_dones = np.concatenate([item[4][:, 0] for item in batches])
    events = np.concatenate([info["events"][:, 0] for info in infos])
    prediction = _one_step(ensemble, stacks, actions, device)
    calibration = calibration_metrics(
        prediction, (real_frames, real_rewards, real_dones), events)
    calibration.pop("_arrays")
    source_calibration = {}
    for name, slc in (
            ("natural", slice(0, sample_count)),
            ("event_balanced", slice(sample_count, 2 * sample_count))):
        source_prediction = {
            key: value[slc] for key, value in prediction.items()
        }
        source_metrics = calibration_metrics(
            source_prediction,
            (real_frames[slc], real_rewards[slc], real_dones[slc]),
            events[slc])
        source_metrics.pop("_arrays")
        source_calibration[name] = source_metrics
    calibration["by_sample_source"] = source_calibration

    horizon = args.horizon or config.SCALES[args.scale]["rollout_eval_h"]
    rollout, rollout_frames, rollout_rewards = natural_rollout(
        ensemble, data, horizon, min(16, sample_count), device)
    lockstep = policy_lockstep(ensemble, policy, horizon, args.seed + 1, device)
    forced, forced_detail, forced_frames, forced_rewards = forced_scenarios(
        ensemble, device)
    invariant = invariant_metrics(
        np.asarray(rollout_frames + lockstep["frames"] + forced_frames),
        np.asarray(rollout_rewards + lockstep["rewards"] + forced_rewards))

    metrics = {
        "natural_rollout_mse": rollout,
        "policy_lockstep_mse": lockstep["mse"],
        "forced_scenario_mse": forced,
        "reward_mae": calibration["reward_mae"],
        "done_brier": calibration["done_brier"],
        "ballless_positive_rate": invariant["ballless_positive_rate"],
        "dream_real_reward_gap": lockstep["reward_gap"],
        "uncertainty_rank_corr": calibration["uncertainty_rank_corr"],
    }
    thresholds = _parse_thresholds(args.scale, args.threshold)
    gates, overall = aggregate_gates(metrics, thresholds)
    diagnostics = stochastic_diagnostics(
        ensemble, stacks[:min(32, len(stacks))],
        actions[:min(32, len(actions))], args.stochastic_samples,
        args.seed, device)
    return {
        "schema_version": 1,
        "tag": args.tag,
        "scale": args.scale,
        "seed": args.seed,
        "headed_checkpoints": list(map(str, args.wm)),
        "policy": str(args.policy) if args.policy else None,
        "data": str(args.data),
        "deterministic_epistemic_prior": True,
        "samples": {"natural": sample_count, "event_balanced": sample_count},
        "measurements": {
            "natural_logged_action_rollout": rollout,
            "current_policy_lockstep": {
                "policy": "checkpoint" if policy else "deterministic_tracker",
                "frame_mse": lockstep["mse"],
                "dream_real_reward_gap": lockstep["reward_gap"],
            },
            "forced_scenarios": forced_detail,
            "reward_done_calibration": calibration,
            "physical_invariants": invariant,
            "uncertainty": {
                "rank_correlation": calibration["uncertainty_rank_corr"],
                "high_error_auroc": calibration["uncertainty_high_error_auroc"],
                "stochastic_diagnostics": diagnostics,
            },
        },
        "thresholds": thresholds,
        "gates": gates,
        "overall_pass": overall,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wm", nargs="+", required=True,
                    help="one or more headed world-model checkpoints")
    ap.add_argument("--policy", default=None,
                    help="optional current policy checkpoint for lockstep actions")
    ap.add_argument("--data", required=True)
    ap.add_argument("--scale", choices=tuple(config.SCALES), default="smoke")
    ap.add_argument("--tag", default="trust")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--stochastic-samples", type=int, default=0)
    ap.add_argument("--threshold", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--out", default=None)
    ap.add_argument("--enforce-trust", action=argparse.BooleanOptionalAction,
                    default=False)
    args = ap.parse_args()
    report = evaluate(args)
    out = Path(args.out) if args.out else \
        config.RESULTS_DIR / f"trust_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print("TRUST_EVAL_" + ("PASS" if report["overall_pass"] else "FAIL"),
          json.dumps({"report": str(out), "overall_pass": report["overall_pass"]}))
    return 1 if args.enforce_trust and not report["overall_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
