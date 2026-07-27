"""Sequential, resumable round-two honest-agent orchestration.

Dry-run is the default. No export stage exists: promotion only writes evidence.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import config  # noqa: E402


def command(backend, module, argv, gpu):
    if backend == "local":
        return [sys.executable, "-m", module, *argv]
    return [
        sys.executable, "-m", "infra.remote", "--module", module,
        "--gpu" if gpu else "--cpu", "--", *argv,
    ]


def build_plan(args):
    root = Path("/vol") if args.backend == "modal" else config.ROOT
    data = root / "data" / f"{args.tag}_targeted"
    checkpoints = root / "checkpoints"
    results = root / "results"
    seeds = list(config.ROUND_TWO["policy_seeds"])
    stages = []

    collect = [
        "--scale", args.scale, "--out", str(data), "--self-check",
        "--mix", "tracker=.1,rare=.2,redteam=.3,stochastic=.3,natural=.1",
        "--redteam-checkpoint", args.redteam,
        "--stochastic-checkpoint", args.stochastic_policy,
        "--world-model-checkpoint", *args.base_wm,
    ]
    if args.transitions is not None:
        collect += ["--transitions", str(args.transitions)]
    stages.append(("collect-targeted",
                   command(args.backend, "scripts.collect_adaptive", collect, True),
                   data / "meta.json"))

    wm_paths = []
    for index, seed in enumerate(seeds):
        path = checkpoints / f"{args.tag}_wm{index}.pt"
        wm_paths.append(path)
        argv = [
            "--scale", args.scale, "--data", str(data), "--v2", "--stochastic",
            "--sampler", "balanced", "--bootstrap", "--seed", str(seed),
            "--tag", f"{args.tag}_wm{index}", "--out", str(path),
        ]
        if args.steps is not None:
            argv += ["--steps", str(args.steps)]
        stages.append((f"train-wm-{index}",
                       command(args.backend, "wm.train", argv, True), path))

    pretrust = results / f"trust_{args.tag}_prepolicy.json"
    stages.append(("trust-before-policy", command(
        args.backend, "scripts.evaluate_trust", [
            "--scale", args.scale, "--data", str(data),
            "--wm", *map(str, wm_paths), "--stochastic-samples", "8",
            "--enforce-trust", "--tag", f"{args.tag}_prepolicy",
            "--out", str(pretrust),
        ], True), pretrust))

    trust_paths, transfer_paths = [], []
    for index, seed in enumerate(seeds):
        policy = checkpoints / f"{args.tag}_policy_seed{seed}.pt"
        dream_log = results / f"dream_train_log_{args.tag}_seed{seed}.json"
        argv = [
            "--scale", args.scale, "--data", str(data),
            "--wm", *map(str, wm_paths), "--algorithm", "ppo",
            "--latent-mode", "sample", "--frame-mode", "mean", "--ball-guard",
            "--seed", str(seed), "--tag", f"{args.tag}_seed{seed}",
            "--out", str(policy),
        ]
        if args.updates is not None:
            argv += ["--updates", str(args.updates)]
        stages.append((f"train-policy-{index}", command(
            args.backend, "agents.train_dream", argv, True), policy))

        transfer = results / f"transfer_{args.tag}_seed{seed}.json"
        transfer_paths.append(transfer)
        stages.append((f"evaluate-policy-{index}", command(
            args.backend, "agents.evaluate", [
                "--scale", args.scale, "--episodes",
                str(config.ROUND_TWO["eval_episodes"]), "--seed", str(seed),
                "--policies", f"dream={policy}", "--gif-steps", "0",
                "--dream-log", str(dream_log), "--out", str(transfer),
            ], False), transfer))

        trust = results / f"trust_{args.tag}_seed{seed}.json"
        trust_paths.append(trust)
        stages.append((f"trust-policy-{index}", command(
            args.backend, "scripts.evaluate_trust", [
                "--scale", args.scale, "--data", str(data),
                "--wm", *map(str, wm_paths), "--policy", str(policy),
                "--seed", str(seed), "--stochastic-samples", "8",
                "--enforce-trust", "--tag", f"{args.tag}_seed{seed}",
                "--out", str(trust),
            ], True), trust))

    promotion = results / f"promotion_{args.tag}.json"
    stages.append(("aggregate-promotion", command(
        args.backend, "scripts.promote_round_two", [
            "--trust", *map(str, trust_paths),
            "--transfer", *map(str, transfer_paths),
            "--out", str(promotion), "--enforce",
        ], False), promotion))
    return stages


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute", action="store_true")
    result.add_argument("--backend", choices=("local", "modal"), default="local")
    result.add_argument("--scale", choices=("smoke", "full"), default="smoke")
    result.add_argument("--tag", default="round_two")
    result.add_argument("--base-wm", nargs="+", required=True)
    result.add_argument("--redteam", required=True)
    result.add_argument("--stochastic-policy", required=True)
    result.add_argument("--transitions", type=int)
    result.add_argument("--steps", type=int)
    result.add_argument("--updates", type=int)
    result.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True)
    return result


def main():
    args = parser().parse_args()
    stages = build_plan(args)
    state_path = config.RESULTS_DIR / f"{args.tag}_round_two_state.json"
    state = json.loads(state_path.read_text()) if (
        args.resume and state_path.exists()) else {"stages": {}}
    for name, argv, artifact in stages:
        if args.resume and state["stages"].get(name, {}).get("ok"):
            print(f"[round-two] SKIP {name}")
            continue
        print(f"[round-two] {'RUN' if args.execute else 'DRY-RUN'} {name}\n"
              f"  {shlex.join(argv)}")
        if not args.execute:
            continue
        completed = subprocess.run(argv, cwd=REPO)
        ok = completed.returncode == 0
        state["stages"][name] = {
            "ok": ok, "artifact": str(artifact), "command": argv}
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
        if not ok:
            return completed.returncode
    print("[round-two] complete" if args.execute else
          "[round-two] dry-run complete; pass --execute to launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
