"""Collect adaptive, contiguous *real* VecPong streams for the data flywheel.

No generated frame is ever stored: frames and targets always come from
``VecPong``. A world model, when supplied, is used only to score one-step
real-vs-predicted disagreement.

Event codes (events.npy, uint8; one mutually-exclusive label per transition):
  0 ordinary
  1 right-paddle hit
  2 agent scores
  3 agent concedes
  4 real episode done by time truncation
  5 serve or near-terminal state
  6 forced stream boundary (not a game event)

The final ``natural`` segment is plain tracker collection and is marked in
meta.json. Per-environment streams and every forced boundary are also recorded.

Examples:
  python -m scripts.collect_adaptive --scale smoke
  python -m scripts.collect_adaptive --scale smoke --transitions 512 --self-check
  python -m scripts.collect_adaptive --scale full --world-model-checkpoint CKPT
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import config
from pong.env import VecPong, scripted_action
from wm.data import EVENT_CODES, EVENT_NAMES


def _parse_mix(text):
    if text is None:
        return dict(config.FLYWHEEL["composition"])
    result = {}
    for item in text.split(","):
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in config.FLYWHEEL["composition"]:
            raise ValueError(f"unknown collection policy {name!r}")
        result[name] = float(value)
    if not result or any(v < 0 for v in result.values()) or sum(result.values()) <= 0:
        raise ValueError("mix weights must be non-negative with a positive sum")
    return result


def _allocate(total, mix):
    """Largest-remainder allocation, with natural kept as the final segment."""
    names = [name for name in mix if name != "natural"]
    if "natural" in mix:
        names.append("natural")
    weight = np.asarray([mix[name] for name in names], np.float64)
    raw = total * weight / weight.sum()
    count = np.floor(raw).astype(np.int64)
    for i in np.argsort(-(raw - count))[:total - int(count.sum())]:
        count[i] += 1
    return [(name, int(n)) for name, n in zip(names, count) if n]


def _load_optional_models(args, device, needed_policies):
    policies = {}

    paths = {
        "baseline": Path(args.baseline_checkpoint),
        "dream": Path(args.dream_checkpoint),
        "redteam": Path(getattr(
            args, "redteam_checkpoint", config.CKPT_DIR / "redteam_agent.pt")),
    }
    for name, path in paths.items():
        if name not in needed_policies:
            continue
        if path.exists():
            try:
                if device is None:
                    raise RuntimeError("torch/device unavailable")
                from agents.policy import load_policy
                policies[name] = load_policy(path, device).eval()
                print(f"[collect_adaptive] loaded {name} policy: {path}", flush=True)
            except Exception as exc:
                print(f"[collect_adaptive] warning: cannot load {name} policy "
                      f"{path}: {exc}; using tracker fallback", flush=True)
        else:
            print(f"[collect_adaptive] warning: {name} checkpoint absent at "
                  f"{path}; using tracker fallback", flush=True)

    wm = None
    raw_wm_paths = getattr(args, "world_model_checkpoint", None) or []
    if isinstance(raw_wm_paths, (str, Path)):
        raw_wm_paths = [raw_wm_paths]
    wm_paths = [Path(item) for group in raw_wm_paths
                for item in (group if isinstance(group, (list, tuple)) else [group])]
    existing_wm_paths = [path for path in wm_paths if path.exists()]
    if wm_paths and len(existing_wm_paths) == len(wm_paths):
        try:
            if device is None:
                raise RuntimeError("torch/device unavailable")
            import torch
            from wm.model import load_wm_ensemble
            for path in wm_paths:
                state = torch.load(path, map_location="cpu", weights_only=True)["model"]
                if not (any(key.startswith("reward_head.") for key in state)
                        and any(key.startswith("done_head.") for key in state)):
                    raise ValueError(f"{path} has no trained reward/done heads")
            wm = load_wm_ensemble(wm_paths, device, with_heads=True).eval()
            print(f"[collect_adaptive] loaded headed world models: {wm_paths}",
                  flush=True)
        except Exception as exc:
            print(f"[collect_adaptive] warning: cannot load world models "
                  f"{wm_paths}: {exc}; priorities will be event-only", flush=True)
    elif wm_paths:
        missing = [path for path in wm_paths if not path.exists()]
        print(f"[collect_adaptive] warning: world-model checkpoints absent: "
              f"{missing}; priorities will be event-only", flush=True)
    return policies, wm


def _actions(kind, env, stacks, rng, policies, device):
    if kind in ("tracker", "natural"):
        return scripted_action(env.right_y, env.ball_y,
                               config.COLLECT["eps_random"], rng)
    if kind == "random":
        return rng.integers(0, config.N_ACTIONS, size=env.n_envs)
    if kind == "rare":
        # Engineer misses while the ball approaches the agent; track while it
        # travels away. This enriches paddle-edge, concede and terminal states.
        track = scripted_action(env.right_y, env.ball_y, 0.05, rng)
        pc = env.right_y + config.ENV["paddle_h"] / 2.0
        bc = env.ball_y + config.ENV["ball_size"] / 2.0
        away = np.where(bc >= pc, 1, 2)
        return np.where(env.ball_vx > 0, away, track).astype(np.int64)
    if kind in policies:
        import torch
        with torch.no_grad():
            x = torch.from_numpy(stacks).to(device).float() / 255.0
            logits, _ = policies[kind](x)
            return torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
    # Missing policy checkpoints must not make smoke/local collection unusable.
    return scripted_action(env.right_y, env.ball_y,
                           config.COLLECT["eps_random"], rng)


def _wm_prediction(wm, stacks, actions, device):
    if wm is None:
        return None
    import torch
    with torch.no_grad():
        x = torch.from_numpy(stacks).to(device).float() / 255.0
        a = torch.from_numpy(np.asarray(actions)).to(device, torch.long)
        prediction = wm.predict(x, a)
        return {
            "frame": prediction.mean_frame[:, 0].cpu().numpy(),
            "reward": prediction.mean_reward.cpu().numpy(),
            "done": prediction.mean_done_prob.cpu().numpy(),
            "frame_disagreement": prediction.frame_mse.cpu().numpy(),
            "reward_disagreement": prediction.reward_std.cpu().numpy(),
            "done_disagreement": prediction.done_disagreement.cpu().numpy(),
        }


def _priority_bonus(prediction, next_frame, reward, done):
    """WM ensemble uncertainty plus one-step real-vs-dream mismatch."""
    if prediction is None:
        return np.zeros(len(done), np.float32)
    real_frame = next_frame.astype(np.float32) / 255.0
    ensemble = (prediction["frame_disagreement"]
                + prediction["reward_disagreement"]
                + prediction["done_disagreement"])
    frame_mismatch = np.abs(prediction["frame"] - real_frame).mean((1, 2))
    reward_mismatch = np.abs(prediction["reward"] - reward)
    done_mismatch = np.abs(prediction["done"] - done.astype(np.float32))
    flywheel = config.FLYWHEEL
    return (
        flywheel["disagreement_weight"] * ensemble
        + flywheel["frame_mismatch_weight"] * frame_mismatch
        + flywheel["reward_mismatch_weight"] * reward_mismatch
        + flywheel["done_mismatch_weight"] * done_mismatch
    ).astype(np.float32)


def _event_labels(env_t, ball_x, info, done):
    n = len(done)
    event = np.full(n, EVENT_CODES["ordinary"], np.uint8)
    near = ((env_t == 0) | (ball_x < 8.0) |
            (ball_x > config.ENV["W"] - 10.0))
    event[near] = EVENT_CODES["serve_near_terminal"]
    event[done & (info["point"] == 0)] = EVENT_CODES["done_truncation"]
    event[info["paddle_hit"]] = EVENT_CODES["hit"]
    event[info["point"] == 1] = EVENT_CODES["score"]
    event[info["point"] == -1] = EVENT_CODES["concede"]
    return event


def collect(args):
    total = args.transitions if args.transitions is not None \
        else config.SCALES[args.scale]["transitions"]
    if total <= 0:
        raise ValueError("--transitions must be positive")
    out = Path(args.out) if args.out else config.DATA_DIR / f"{args.scale}_adaptive"
    out.mkdir(parents=True, exist_ok=True)
    mix = _parse_mix(args.mix)
    allocation = _allocate(total, mix)
    device = None
    if {"baseline", "dream", "redteam"} & set(mix) or args.world_model_checkpoint:
        try:
            device = config.get_device()
        except ImportError:
            print("[collect_adaptive] warning: torch unavailable; policy/WM "
                  "segments will gracefully fall back", flush=True)
    policies, wm = _load_optional_models(args, device, set(mix))
    E = config.ENV

    frames = np.lib.format.open_memmap(
        out / "frames.npy", mode="w+", dtype=np.uint8,
        shape=(total, E["H"], E["W"]))
    actions = np.lib.format.open_memmap(
        out / "actions.npy", mode="w+", dtype=np.uint8, shape=(total,))
    rewards = np.lib.format.open_memmap(
        out / "rewards.npy", mode="w+", dtype=np.float32, shape=(total,))
    dones = np.lib.format.open_memmap(
        out / "dones.npy", mode="w+", dtype=bool, shape=(total,))
    events = np.lib.format.open_memmap(
        out / "events.npy", mode="w+", dtype=np.uint8, shape=(total,))
    priorities = np.lib.format.open_memmap(
        out / "priorities.npy", mode="w+", dtype=np.float32, shape=(total,))

    segments, streams = [], []
    cursor = 0
    true_dones = 0
    event_counts = np.zeros(max(EVENT_NAMES) + 1, np.int64)
    priority_by_code = np.asarray([
        config.FLYWHEEL["event_priority"][EVENT_NAMES[i]]
        for i in range(max(EVENT_NAMES) + 1)], np.float32)
    started = time.perf_counter()

    for segment_index, (kind, segment_n) in enumerate(allocation):
        # Tiny smoke overrides still need streams long enough for a four-frame
        # stack and the configured k-step training window.
        min_stream = config.FRAME_STACK + config.WM["unroll_k"] + 1
        n_envs = min(config.COLLECT["n_envs"],
                     max(1, segment_n // min_stream))
        quota = np.full(n_envs, segment_n // n_envs, np.int64)
        quota[:segment_n % n_envs] += 1
        offsets = cursor + np.concatenate([[0], np.cumsum(quota)[:-1]])
        env = VecPong(n_envs, seed=args.seed + 1009 * segment_index)
        rng = np.random.default_rng(args.seed + 7919 * (segment_index + 1))
        cur = env.reset()
        stacks = np.repeat(cur[:, None], config.FRAME_STACK, axis=1)
        actual_kind = kind if kind not in ("baseline", "dream", "redteam") or kind in policies \
            else "tracker_fallback"

        for step in range(int(quota.max())):
            active = step < quota
            action = _actions(kind, env, stacks, rng, policies, device)
            prediction = _wm_prediction(wm, stacks, action, device)
            env_t, ball_x = env.t.copy(), env.ball_x.copy()
            nxt, reward, done, info = env.step(action)
            label = _event_labels(env_t, ball_x, info, done)
            idx = (offsets + step)[active]
            frames[idx] = cur[active]
            actions[idx] = np.asarray(action[active], np.uint8)
            rewards[idx] = reward[active]
            dones[idx] = done[active]
            events[idx] = label[active]
            p = priority_by_code[label]
            if prediction is not None:
                p = p + _priority_bonus(prediction, nxt, reward, done)
            priorities[idx] = np.maximum(
                p[active], config.FLYWHEEL["priority_floor"]).astype(np.float32)
            true_dones += int(done[active].sum())
            event_counts += np.bincount(label[active], minlength=len(event_counts))
            stacks = np.concatenate([stacks[:, 1:], nxt[:, None]], axis=1)
            if done.any():
                stacks[done] = nxt[done, None]
            cur = nxt

        # Isolate every per-env contiguous stream. Keep synthetic boundaries
        # distinct from real truncations so balanced sampling cannot overfit them.
        boundary = offsets + quota - 1
        old = np.asarray(events[boundary]).copy()
        event_counts -= np.bincount(old, minlength=len(event_counts))
        event_counts[EVENT_CODES["boundary"]] += len(boundary)
        dones[boundary] = True
        events[boundary] = EVENT_CODES["boundary"]
        priorities[boundary] = priority_by_code[EVENT_CODES["boundary"]]
        for env_i, (start, length) in enumerate(zip(offsets, quota)):
            streams.append({
                "start": int(start), "end": int(start + length),
                "segment": segment_index, "env": env_i,
            })
        segments.append({
            "name": kind, "actual_policy": actual_kind,
            "start": cursor, "end": cursor + segment_n,
            "T": segment_n, "natural_holdout": kind == "natural",
        })
        cursor += segment_n
        print(f"[collect_adaptive] {kind}: T={segment_n} policy={actual_kind}",
              flush=True)

    for array in (frames, actions, rewards, dones, events, priorities):
        array.flush()
    elapsed = time.perf_counter() - started
    meta = {
        "seed": args.seed, "scale": args.scale, "T": total,
        "real_frames_only": True,
        "n_episodes": int(np.asarray(dones).sum()),
        "n_true_dones": true_dones,
        "n_forced_boundary_dones": len(streams),
        "mix": mix, "segments": segments, "streams": streams,
        "natural_holdout_ranges": [
            [s["start"], s["end"]] for s in segments if s["natural_holdout"]],
        "event_codes": {str(k): v for k, v in EVENT_NAMES.items()},
        "event_counts": {
            EVENT_NAMES[i]: int(event_counts[i]) for i in range(len(event_counts))},
        "priorities": {
            "world_model_checkpoint": args.world_model_checkpoint,
            "world_model_loaded": wm is not None,
            "event_priority": config.FLYWHEEL["event_priority"],
            "disagreement_weight": config.FLYWHEEL["disagreement_weight"],
            "frame_mismatch_weight": config.FLYWHEEL["frame_mismatch_weight"],
            "reward_mismatch_weight": config.FLYWHEEL["reward_mismatch_weight"],
            "done_mismatch_weight": config.FLYWHEEL["done_mismatch_weight"],
        },
        "env_constants": dict(config.ENV),
        "collect": dict(config.COLLECT),
        "elapsed_sec": elapsed,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    del frames, actions, rewards, dones, events, priorities
    return out, meta


def _self_check(out, meta):
    arrays = {
        name: np.load(out / f"{name}.npy", mmap_mode="r")
        for name in ("frames", "actions", "rewards", "dones", "events", "priorities")
    }
    T = meta["T"]
    assert all(len(a) == T for a in arrays.values())
    assert arrays["frames"].dtype == np.uint8
    assert arrays["events"].dtype == np.uint8
    assert arrays["priorities"].dtype == np.float32
    assert np.isfinite(arrays["priorities"]).all() and (arrays["priorities"] > 0).all()
    assert arrays["dones"].dtype == bool
    for stream in meta["streams"]:
        assert arrays["dones"][stream["end"] - 1]
        assert arrays["events"][stream["end"] - 1] == EVENT_CODES["boundary"]
    natural = meta["natural_holdout_ranges"]
    if "natural" in meta["mix"] and meta["mix"]["natural"] > 0:
        assert natural and natural[-1][1] == T
    print(f"COLLECT_ADAPTIVE_SELF_CHECK_OK T={T} streams={len(meta['streams'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", default="smoke", choices=list(config.SCALES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--transitions", type=int, default=None,
                    help="override scale transition count (useful for tiny smoke checks)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--mix", default=None,
                    help="comma-separated weights, e.g. tracker=.5,rare=.3,natural=.2")
    ap.add_argument("--baseline-checkpoint",
                    default=str(config.CKPT_DIR / "ppo_baseline.pt"))
    ap.add_argument("--dream-checkpoint",
                    default=str(config.CKPT_DIR / "dream_agent.pt"))
    ap.add_argument("--redteam-checkpoint",
                    default=str(config.CKPT_DIR / "redteam_agent.pt"))
    ap.add_argument("--world-model-checkpoint", nargs="+", action="append", default=None,
                    help="headed WM checkpoint(s); repeat for an ensemble")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    config.seed_everything(args.seed)
    out, meta = collect(args)
    if args.self_check:
        _self_check(out, meta)
    print(f"COLLECT_ADAPTIVE_OK T={meta['T']} real_only=True -> {out}")


if __name__ == "__main__":
    main()
