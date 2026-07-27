"""Safe sequential driver for adaptive-data world-model flywheel experiments.

Dry-run is the default. Add ``--execute`` to run stages, for example:

  python -m scripts.run_flywheel --scale smoke --mode combined
  python -m scripts.run_flywheel --execute --backend local --scale smoke
  python -m scripts.run_flywheel --execute --backend modal --scale full

Every command runs synchronously before the next starts. This intentionally
avoids concurrent Modal app creation and its rate limits.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import config  # noqa: E402


ABLATIONS = {
    "balanced-only": {"members": 1, "sampler": "balanced", "pessimism": False},
    "ensemble-only": {"members": 3, "sampler": "natural", "pessimism": False},
    # Pessimism needs multiple predictors to estimate epistemic uncertainty.
    "pessimism-only": {"members": 3, "sampler": "natural", "pessimism": True},
    "combined": {"members": 3, "sampler": "balanced", "pessimism": True},
}


def _paths(backend: str, name: str):
    root = Path("/vol") if backend == "modal" else config.ROOT
    return root / "data" / f"{name}_data", root / "checkpoints"


def _command(backend: str, module: str, argv: list[str], gpu: bool):
    if backend == "local":
        return [sys.executable, "-m", module, *argv]
    return [
        sys.executable, "-m", "infra.remote", "--module", module,
        "--gpu" if gpu else "--cpu", "--", *argv,
    ]


def _artifact_exists(path: Path, backend: str):
    # Remote completion is tracked in the local state file; probing Modal here
    # would itself create an app and defeat dry-run/rate-limit safety.
    return backend == "local" and path.exists()


def _write_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true",
                    help="actually launch commands (default prints only)")
    ap.add_argument("--backend", choices=("local", "modal"), default="local")
    ap.add_argument("--scale", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--mode", action="append", choices=tuple(ABLATIONS),
                    help="ablation mode; repeat to run several (default: combined)")
    ap.add_argument("--tag", default="flywheel")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="skip stages marked successful in the state file")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction,
                    default=True, help="skip existing local artifacts")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--base-wm", default=None,
                    help="optional prior WM used only to score real-data priorities")
    ap.add_argument("--transitions", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--redteam-updates", type=int, default=None)
    ap.add_argument("--second-transitions", type=int, default=None)
    ap.add_argument("--final-retrain", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="retrain the ensemble on red-team-enriched real data")
    ap.add_argument("--stochastic", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="after deterministic stages, run a gated stochastic comparison")
    ap.add_argument("--enforce-trust", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="stop when a trust report fails (default records only)")
    ap.add_argument("--latent-dim", type=int, default=config.WM["latent_dim"])
    ap.add_argument("--kl-coef", type=float, default=config.WM["kl_coef"])
    ap.add_argument("--ball-guard", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    modes = args.mode or ["combined"]
    state_path = config.RESULTS_DIR / f"{args.tag}_flywheel_state.json"
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"tag": args.tag, "backend": args.backend, "scale": args.scale,
                 "stages": {}}

    for mode in modes:
        spec = ABLATIONS[mode]
        run_name = f"{args.tag}_{mode.replace('-', '_')}"
        data_dir, ckpt_dir = _paths(args.backend, run_name)
        enriched_data_dir = data_dir.parent / f"{run_name}_enriched_data"
        seeds = list(config.FLYWHEEL["ensemble_seeds"])[:spec["members"]]
        if len(seeds) < spec["members"]:
            raise ValueError("config.FLYWHEEL ensemble_seeds is shorter than ensemble_size")

        collect_argv = ["--scale", args.scale, "--out", str(data_dir),
                        "--seed", str(config.SEED), "--self-check",
                        "--mix", ("tracker=.30,baseline=.20,dream=.15,"
                                  "random=.10,rare=.15,natural=.10")]
        if args.transitions is not None:
            collect_argv += ["--transitions", str(args.transitions)]
        if args.base_wm:
            collect_argv += ["--world-model-checkpoint", args.base_wm]
        stages = [(
            f"{run_name}:collect",
            _command(args.backend, "scripts.collect_adaptive", collect_argv, False),
            data_dir / "meta.json",
        )]

        wm_paths = []
        for member, seed in enumerate(seeds):
            wm_tag = f"{run_name}_wm{member}"
            wm_path = ckpt_dir / f"{wm_tag}.pt"
            wm_paths.append(wm_path)
            train_argv = [
                "--scale", args.scale, "--data", str(data_dir), "--v2",
                "--tag", wm_tag, "--seed", str(seed),
                "--sampler", spec["sampler"], "--bootstrap",
                "--bootstrap-frac", str(config.FLYWHEEL["bootstrap_frac"]),
            ]
            if args.steps is not None:
                train_argv += ["--steps", str(args.steps)]
            stages.append((
                f"{run_name}:wm{member}",
                _command(args.backend, "wm.train", train_argv, True),
                wm_path,
            ))

        redteam_path = ckpt_dir / f"{run_name}_redteam.pt"
        redteam_argv = [
            "--scale", args.scale, "--data", str(data_dir),
            "--out", str(redteam_path), "--tag", run_name,
            "--wm", *map(str, wm_paths), "--seed", str(config.SEED),
        ]
        if args.redteam_updates is not None:
            redteam_argv += ["--updates", str(args.redteam_updates)]
        elif args.updates is not None:
            redteam_argv += ["--updates", str(args.updates)]
        stages.append((
            f"{run_name}:redteam",
            _command(args.backend, "agents.train_redteam", redteam_argv, True),
            redteam_path,
        ))

        enriched_argv = [
            "--scale", args.scale, "--out", str(enriched_data_dir),
            "--seed", str(config.SEED + 1), "--self-check",
            "--mix", ("tracker=.15,baseline=.10,dream=.10,redteam=.35,"
                      "random=.05,rare=.15,natural=.10"),
            "--redteam-checkpoint", str(redteam_path),
            "--world-model-checkpoint", *map(str, wm_paths),
        ]
        second_transitions = (args.second_transitions if args.second_transitions
                              is not None else args.transitions)
        if second_transitions is not None:
            enriched_argv += ["--transitions", str(second_transitions)]
        stages.append((
            f"{run_name}:collect-redteam",
            # Priority scoring runs every real transition through the full WM
            # ensemble. Use a GPU remotely; CPU collection is only appropriate
            # for the initial pass that has no world-model inference.
            _command(args.backend, "scripts.collect_adaptive", enriched_argv, True),
            enriched_data_dir / "meta.json",
        ))

        dream_wm_paths = wm_paths
        if args.final_retrain:
            dream_wm_paths = []
            for member, seed in enumerate(seeds):
                wm_tag = f"{run_name}_final_wm{member}"
                wm_path = ckpt_dir / f"{wm_tag}.pt"
                dream_wm_paths.append(wm_path)
                train_argv = [
                    "--scale", args.scale, "--data", str(enriched_data_dir), "--v2",
                    "--tag", wm_tag, "--seed", str(seed),
                    "--sampler", spec["sampler"], "--bootstrap",
                    "--bootstrap-frac", str(config.FLYWHEEL["bootstrap_frac"]),
                ]
                if args.steps is not None:
                    train_argv += ["--steps", str(args.steps)]
                stages.append((
                    f"{run_name}:final-wm{member}",
                    _command(args.backend, "wm.train", train_argv, True),
                    wm_path,
                ))

        dream_path = ckpt_dir / f"{run_name}_dream.pt"
        dream_argv = [
            "--scale", args.scale, "--data", str(enriched_data_dir),
            "--out", str(dream_path),
            "--tag", run_name,
            "--wm", *map(str, dream_wm_paths), "--seed", str(config.SEED),
            "--frame-mode", config.DREAM["frame_mode"],
        ]
        if args.ball_guard:
            dream_argv.append("--ball-guard")
        if args.updates is not None:
            dream_argv += ["--updates", str(args.updates)]
        if not spec["pessimism"]:
            dream_argv += [
                "--reward-disagreement-coef", "0",
                "--frame-disagreement-coef", "0",
                "--done-disagreement-coef", "0",
                "--uncertainty-continuation-coef", "0",
            ]
        stages.append((
            f"{run_name}:dream",
            _command(args.backend, "agents.train_dream", dream_argv, True),
            dream_path,
        ))

        results_dir = (Path("/vol") if args.backend == "modal"
                       else config.ROOT) / "results"
        transfer_path = results_dir / f"transfer_{run_name}.json"
        stages.append((
            f"{run_name}:evaluate",
            _command(args.backend, "agents.evaluate", [
                "--scale", args.scale,
                "--policies", f"dream={dream_path}",
                "--tag", run_name,
                "--dream-log", str(results_dir / f"dream_train_log_{run_name}.json"),
                "--out", str(transfer_path),
            ], False),
            transfer_path,
        ))
        trust_path = results_dir / f"trust_{run_name}.json"
        trust_argv = [
            "--scale", args.scale, "--data", str(enriched_data_dir),
            "--wm", *map(str, dream_wm_paths),
            "--policy", str(dream_path), "--tag", run_name,
            "--seed", str(config.SEED), "--out", str(trust_path),
        ]
        if args.enforce_trust:
            trust_argv.append("--enforce-trust")
        stages.append((
            f"{run_name}:trust",
            _command(args.backend, "scripts.evaluate_trust", trust_argv, True),
            trust_path,
        ))

        # Optional final ablation: train a single CVAE on exactly the same
        # enriched data and run dreams with sampled latents. This is deliberately
        # gated because a full stochastic ensemble would multiply final cost.
        if args.stochastic:
            stochastic_tag = f"{run_name}_stochastic_wm"
            stochastic_path = ckpt_dir / f"{stochastic_tag}.pt"
            stochastic_train_argv = [
                "--scale", args.scale, "--data", str(enriched_data_dir), "--v2",
                "--stochastic", "--latent-dim", str(args.latent_dim),
                "--kl-coef", str(args.kl_coef), "--tag", stochastic_tag,
                "--seed", str(seeds[0]), "--sampler", spec["sampler"],
                "--bootstrap", "--bootstrap-frac",
                str(config.FLYWHEEL["bootstrap_frac"]),
            ]
            if args.steps is not None:
                stochastic_train_argv += ["--steps", str(args.steps)]
            stages.append((
                f"{run_name}:compare-stochastic-wm",
                _command(args.backend, "wm.train", stochastic_train_argv, True),
                stochastic_path,
            ))
            stochastic_dream_path = ckpt_dir / f"{run_name}_stochastic_dream.pt"
            stochastic_dream_argv = [
                "--scale", args.scale, "--data", str(enriched_data_dir),
                "--out", str(stochastic_dream_path),
                "--tag", f"{run_name}_stochastic", "--wm", str(stochastic_path),
                "--seed", str(config.SEED), "--frame-mode", "mean",
                "--latent-mode", "sample",
            ]
            if args.ball_guard:
                stochastic_dream_argv.append("--ball-guard")
            if args.updates is not None:
                stochastic_dream_argv += ["--updates", str(args.updates)]
            stages.append((
                f"{run_name}:compare-stochastic-dream",
                _command(args.backend, "agents.train_dream",
                         stochastic_dream_argv, True),
                stochastic_dream_path,
            ))
            stochastic_run_name = f"{run_name}_stochastic"
            stochastic_transfer_path = (
                results_dir / f"transfer_{stochastic_run_name}.json")
            stages.append((
                f"{run_name}:compare-stochastic-evaluate",
                _command(args.backend, "agents.evaluate", [
                    "--scale", args.scale,
                    "--policies", f"dream={stochastic_dream_path}",
                    "--tag", stochastic_run_name,
                    "--dream-log", str(
                        results_dir / f"dream_train_log_{stochastic_run_name}.json"),
                    "--out", str(stochastic_transfer_path),
                ], False),
                stochastic_transfer_path,
            ))
            stochastic_trust_path = (
                results_dir / f"trust_{stochastic_run_name}.json")
            stochastic_trust_argv = [
                "--scale", args.scale, "--data", str(enriched_data_dir),
                "--wm", str(stochastic_path),
                "--policy", str(stochastic_dream_path),
                "--tag", stochastic_run_name, "--seed", str(config.SEED),
                "--stochastic-samples", "4",
                "--out", str(stochastic_trust_path),
            ]
            if args.enforce_trust:
                stochastic_trust_argv.append("--enforce-trust")
            stages.append((
                f"{run_name}:compare-stochastic-trust",
                _command(args.backend, "scripts.evaluate_trust",
                         stochastic_trust_argv, True),
                stochastic_trust_path,
            ))

        for stage_name, cmd, artifact in stages:
            completed = state["stages"].get(stage_name, {}).get("ok", False)
            if (args.resume and completed) or (
                    args.skip_existing and _artifact_exists(artifact, args.backend)):
                print(f"[flywheel] SKIP {stage_name} -> {artifact}", flush=True)
                continue
            print(f"[flywheel] {'RUN' if args.execute else 'DRY-RUN'} "
                  f"{stage_name}\n  {shlex.join(cmd)}", flush=True)
            if not args.execute:
                continue
            result = None
            for attempt in range(1, args.retries + 2):
                started = time.time()
                result = subprocess.run(cmd, cwd=REPO)
                if result.returncode == 0:
                    break
                if attempt <= args.retries:
                    delay = 15 * attempt
                    print(f"[flywheel] retry {stage_name} in {delay}s", flush=True)
                    time.sleep(delay)
            ok = result is not None and result.returncode == 0
            state["stages"][stage_name] = {
                "ok": ok, "artifact": str(artifact), "command": cmd,
                "finished_at": time.time(),
            }
            _write_state(state_path, state)
            if not ok:
                print(f"[flywheel] STOP: {stage_name} failed", file=sys.stderr)
                return 1

    print("[flywheel] complete" if args.execute else
          "[flywheel] dry-run complete; pass --execute to launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
