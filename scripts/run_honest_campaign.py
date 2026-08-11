"""Budget-aware, sequential controller for the imagination-only campaign.

The controller never consumes final results for acquisition or selection. It
runs the immutable final panel at most once, after development-only promotion.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import config  # noqa: E402

STOP_ERROR = re.compile(
    r"(insufficient.*credit|quota.*exceed|payment required|not authenticated|"
    r"unauthorized|credit.*exhaust)", re.IGNORECASE)


def modal_credit_telemetry():
    """Return (remaining credit, source, status) without guessing values."""
    environment = os.environ.get("MODAL_CREDIT_BALANCE_USD")
    if environment is not None:
        try:
            return float(environment), "MODAL_CREDIT_BALANCE_USD", "available"
        except ValueError:
            return None, "MODAL_CREDIT_BALANCE_USD", "invalid"
    result = subprocess.run(
        [sys.executable, "-m", "modal", "billing", "summary", "--json"],
        cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        return None, "modal billing summary --json", "authentication_failed"
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None, "modal billing summary --json", "unparseable"
    # Billing schemas can add fields. Only explicit remaining-credit/balance
    # names are accepted; current spend is never misrepresented as balance.
    queue = [payload]
    accepted = {
        "remaining_credits", "remaining_credit", "credit_balance",
        "remaining_balance", "available_credits",
    }
    while queue:
        item = queue.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key.lower() in accepted and isinstance(value, (int, float)):
                    return float(value), "modal billing summary --json", "available"
                queue.append(value)
        elif isinstance(item, list):
            queue.extend(item)
    return None, "modal billing summary --json", "balance_field_unavailable"


def fetch_volume(remote_path, local_path):
    return subprocess.run([
        sys.executable, "-m", "infra.remote", "--fetch",
        str(remote_path).removeprefix("/vol/"), str(local_path),
    ], cwd=REPO)


def remote_command(module, argv, gpu=True):
    return [
        sys.executable, "-m", "infra.remote", "--module", module,
        "--gpu" if gpu else "--cpu", "--", *argv,
    ]


def acquisition_mix(previous_selection):
    if not previous_selection:
        return config.CAMPAIGN["acquisition_mix_default"]
    failed = {
        gate for candidate in previous_selection.get("candidates", [])
        for gate in candidate.get("failed_trust_gates", [])
    }
    # Prefer the binding-repair mix whenever latent utilization failed.
    # The old collapse mix (stochastic=.45) repeatedly made ballless worse
    # without clearing the utilization gate.
    if "stochastic_latent_utilization" in failed:
        return config.CAMPAIGN["acquisition_mix_binding_repair"]
    if "ballless_positive_rate" in failed:
        return config.CAMPAIGN["acquisition_mix_calibration"]
    if any(name.startswith("stochastic_") for name in failed):
        return config.CAMPAIGN["acquisition_mix_collapse"]
    if {"reward_mae", "done_brier", "dream_real_reward_gap"} & failed:
        return config.CAMPAIGN["acquisition_mix_calibration"]
    if "uncertainty_rank_corr" in failed:
        return config.CAMPAIGN["acquisition_mix_uncertainty"]
    return config.CAMPAIGN["acquisition_mix_default"]


def round_plan(args, round_index, mix):
    root = Path("/vol")
    tag = f"{args.tag}_r{round_index}"
    data = root / "data" / tag
    checkpoints = root / "checkpoints"
    results = root / "results"
    acquisition_seed = config.ROUND_TWO["acquisition_seeds"][round_index]
    stages = []

    collect = [
        "--scale", "full", "--out", str(data), "--self-check",
        "--seed", str(acquisition_seed), "--mix", mix,
        "--redteam-checkpoint", args.redteam,
        "--stochastic-checkpoint", args.stochastic_policy,
        "--world-model-checkpoint", *args.base_wm,
        "--transitions", str(args.transitions),
    ]
    stages.append(("collect", remote_command(
        "scripts.collect_adaptive", collect, gpu=True), data / "meta.json", True))

    wm_paths = []
    for index, seed in enumerate(config.ROUND_TWO["wm_seeds"]):
        path = checkpoints / f"{tag}_wm{index}.pt"
        wm_paths.append(path)
        stages.append((f"wm-{index}", remote_command("wm.train", [
            "--scale", "full", "--data", str(data), "--out", str(path),
            "--tag", f"{tag}_wm{index}", "--v2", "--stochastic",
            "--sampler", "balanced", "--bootstrap", "--seed", str(seed),
            "--steps", str(args.steps),
            "--max-seconds", str(config.CAPS["wm_train"]),
        ]), path, True))

    pretrust = results / f"trust_{tag}_prepolicy.json"
    stages.append(("pretrust", remote_command("scripts.evaluate_trust", [
        "--scale", "full", "--data", str(data), "--wm", *map(str, wm_paths),
        "--stochastic-samples", "8",
        "--out", str(pretrust), "--tag", f"{tag}_prepolicy",
    ]), pretrust, True))

    policies, trusts, developments = [], [], []
    for index, seed in enumerate(config.ROUND_TWO["policy_seeds"]):
        policy = checkpoints / f"{tag}_ppo_seed{seed}.pt"
        policies.append(policy)
        stages.append((f"policy-{index}", remote_command("agents.train_dream", [
            "--scale", "full", "--data", str(data), "--wm", *map(str, wm_paths),
            "--algorithm", "ppo", "--latent-mode", "sample", "--ball-guard",
            "--seed", str(seed), "--updates", str(args.updates),
            "--tag", f"{tag}_ppo_seed{seed}", "--out", str(policy),
        ]), policy, True))
        trust = results / f"trust_{tag}_candidate{index}.json"
        trusts.append(trust)
        stages.append((f"policy-trust-{index}", remote_command(
            "scripts.evaluate_trust", [
                "--scale", "full", "--data", str(data),
                "--wm", *map(str, wm_paths), "--policy", str(policy),
                "--seed", str(config.ROUND_TWO["development_seeds"][0]),
                "--stochastic-samples", "8",
                "--out", str(trust), "--tag", f"{tag}_candidate{index}",
            ]), trust, True))
        development = results / f"development_{tag}_candidate{index}.json"
        developments.append(development)
        stages.append((f"development-{index}", remote_command(
            "scripts.evaluate_panel", [
                "--purpose", "development", "--policy", str(policy),
                "--out", str(development),
            ], gpu=False), development, False))

    # One fixed-seed REINFORCE ablation in round zero isolates the PPO change.
    if round_index == 0:
        ablation = checkpoints / f"{tag}_reinforce_ablation.pt"
        stages.append(("reinforce-ablation", remote_command(
            "agents.train_dream", [
                "--scale", "full", "--data", str(data),
                "--wm", *map(str, wm_paths), "--algorithm", "reinforce",
                "--latent-mode", "sample", "--ball-guard",
                "--seed", str(config.ROUND_TWO["policy_seeds"][0]),
                "--updates", str(args.updates), "--tag", f"{tag}_reinforce",
                "--out", str(ablation),
            ]), ablation, True))
        stages.append(("reinforce-development", remote_command(
            "scripts.evaluate_panel", [
                "--purpose", "development", "--policy", str(ablation),
                "--out", str(results / f"development_{tag}_reinforce.json"),
            ], gpu=False), results / f"development_{tag}_reinforce.json", False))

    selection = results / f"selection_{tag}.json"
    stages.append(("selection", remote_command("scripts.select_candidate", [
        "--policies", *map(str, policies), "--trust", *map(str, trusts),
        "--development", *map(str, developments), "--out", str(selection),
    ], gpu=False), selection, False))
    return stages, selection


def preregistration(args):
    return {
        "schema_version": 1,
        "created_at": time.time(),
        "target_mean_win_rate": config.CAMPAIGN["target_final_win_rate"],
        "acquisition_seeds": list(config.ROUND_TWO["acquisition_seeds"]),
        "wm_seeds": list(config.ROUND_TWO["wm_seeds"]),
        "policy_training_seeds": list(config.ROUND_TWO["policy_seeds"]),
        "development_seeds": list(config.ROUND_TWO["development_seeds"]),
        "final_evaluation_seeds": list(
            config.ROUND_TWO["final_evaluation_seeds"]),
        "development_episodes_per_seed": config.ROUND_TWO[
            "development_episodes_per_seed"],
        "final_episodes_per_seed": config.ROUND_TWO["final_episodes_per_seed"],
        "development_promotion_threshold": config.ROUND_TWO[
            "development_promotion_win_rate"],
        "final_panel_max_uses": 1,
        "candidate_rule": "highest development mean, then lowest index",
        "final_results_used_for_tuning": False,
        "max_rounds": args.max_rounds,
        "max_stage_retries": args.retries,
        "real_policy_gradient_steps": 0,
        "modal_credit_balance_usd_at_start": args.credit_balance_usd,
        "modal_credit_telemetry_source": args.credit_telemetry_source,
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute", action="store_true")
    result.add_argument("--tag", default="honest_campaign")
    result.add_argument("--base-wm", nargs="+", required=True)
    result.add_argument("--redteam", required=True)
    result.add_argument("--stochastic-policy", required=True)
    result.add_argument("--transitions", type=int, default=1_000_000)
    result.add_argument("--steps", type=int, default=18_000)
    result.add_argument("--updates", type=int, default=2_500)
    result.add_argument("--max-rounds", type=int,
                        default=config.CAMPAIGN["max_rounds"])
    result.add_argument("--retries", type=int,
                        default=config.CAMPAIGN["max_stage_retries"])
    return result


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def main():
    args = parser().parse_args()
    lock_handle = None
    if args.execute:
        lock_path = config.RESULTS_DIR / f"{args.tag}_campaign.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("w")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[campaign] STOP: another controller holds the campaign lock",
                  file=sys.stderr)
            return 2
        lock_handle.write(f"pid={os.getpid()} started={time.time()}\n")
        lock_handle.flush()
    credit, source, telemetry_status = modal_credit_telemetry()
    args.credit_balance_usd = credit
    args.credit_telemetry_source = source
    manifest_path = config.RESULTS_DIR / f"{args.tag}_campaign_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "preregistration": preregistration(args),
            "rounds": [], "stages": {}, "final_evaluation_uses": 0,
            "modal_credit_telemetry_status": telemetry_status,
            "status": "dry_run" if not args.execute else "running",
        }
        _write(manifest_path, manifest)

    previous_selection = None
    for round_index in range(args.max_rounds):
        mix = acquisition_mix(previous_selection)
        stages, selection_path = round_plan(args, round_index, mix)
        if len(manifest["rounds"]) <= round_index:
            manifest["rounds"].append({
                "index": round_index, "acquisition_mix": mix,
                "selection_report": str(selection_path)})
            _write(manifest_path, manifest)
        round_failure = None
        for name, command, artifact, gpu in stages:
            stage_id = f"round-{round_index}:{name}"
            prior = manifest["stages"].get(stage_id)
            if prior and prior.get("ok"):
                print(f"[campaign] SKIP {stage_id}")
                continue
            print(f"[campaign] {'RUN' if args.execute else 'DRY-RUN'} "
                  f"{stage_id}\n  {shlex.join(command)}")
            if not args.execute:
                continue
            # A live balance value is mandatory before any paid stage.
            if args.credit_balance_usd is None:
                manifest["status"] = "blocked_credit_telemetry_unavailable"
                _write(manifest_path, manifest)
                print("[campaign] STOP: Modal balance telemetry unavailable",
                      file=sys.stderr)
                return 2
            log_path = config.RESULTS_DIR / "campaign_logs" / (
                stage_id.replace(":", "_") + ".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            result = None
            started = time.time()
            for attempt in range(args.retries + 1):
                with log_path.open("a") as log:
                    result = subprocess.run(
                        command, cwd=REPO, stdout=log,
                        stderr=subprocess.STDOUT, text=True)
                if result.returncode == 0:
                    break
                text = log_path.read_text()
                if STOP_ERROR.search(text):
                    manifest["status"] = "budget_or_auth_exhausted"
                    break
            elapsed = time.time() - started
            ok = result is not None and result.returncode == 0
            manifest["stages"][stage_id] = {
                "ok": ok, "command": command, "artifact": str(artifact),
                "log": str(log_path), "wall_seconds": elapsed,
                "modal_gpu_stage": gpu,
                "modal_compute_cost_usd": None,
                "attempts": attempt + 1,
                "returncode": result.returncode if result else None,
            }
            if ok and str(artifact).endswith(".json"):
                evidence_dir = config.RESULTS_DIR / "campaign_evidence"
                evidence_dir.mkdir(parents=True, exist_ok=True)
                local_evidence = evidence_dir / (
                    stage_id.replace(":", "_") + ".json")
                fetched = fetch_volume(artifact, local_evidence)
                if fetched.returncode == 0:
                    payload = json.loads(local_evidence.read_text())
                    manifest["stages"][stage_id]["evidence"] = str(
                        local_evidence)
                    if name == "collect":
                        manifest["stages"][stage_id]["dataset"] = {
                            "T": payload.get("T"),
                            "mix": payload.get("mix"),
                            "event_counts": payload.get("event_counts"),
                            "real_frames_only": payload.get("real_frames_only"),
                        }
                    elif "trust" in name:
                        manifest["stages"][stage_id]["trust"] = {
                            "overall_pass": payload.get("overall_pass"),
                            "gates": payload.get("gates"),
                        }
                        failed_gates = [
                            gate_name for gate_name, gate in
                            payload.get("gates", {}).items()
                            if not gate.get("available", False)
                            or gate.get("pass") is not True
                        ]
                        if payload.get("overall_pass") is not True or failed_gates:
                            round_failure = {
                                "candidates": [{
                                    "failed_trust_gates": failed_gates}]}
                    elif name.startswith("development"):
                        manifest["stages"][stage_id]["development"] = {
                            "mean_win_rate": payload.get("mean_win_rate"),
                            "ci95": payload.get("pooled_win_rate_ci95"),
                            "per_seed": payload.get("per_seed"),
                        }
            _write(manifest_path, manifest)
            if not ok:
                if manifest["status"] == "running":
                    manifest["status"] = "stage_failed"
                    _write(manifest_path, manifest)
                return result.returncode if result else 1
            if round_failure is not None:
                manifest["rounds"][round_index]["stopped_before_policy"] = (
                    name == "pretrust")
                manifest["rounds"][round_index]["failed_trust_gates"] = (
                    round_failure["candidates"][0]["failed_trust_gates"])
                _write(manifest_path, manifest)
                break
        if not args.execute:
            continue
        if round_failure is not None:
            previous_selection = round_failure
            continue
        # Selection retrieval is intentionally explicit. Final evidence is not
        # fetched or inspected until development-only selection succeeds.
        local_selection = config.RESULTS_DIR / f"{args.tag}_r{round_index}_selection.json"
        fetch = fetch_volume(selection_path, local_selection)
        if fetch.returncode != 0:
            manifest["status"] = "selection_fetch_failed"
            _write(manifest_path, manifest)
            return fetch.returncode
        previous_selection = json.loads(local_selection.read_text())
        if previous_selection.get("eligible"):
            if manifest["final_evaluation_uses"] >= 1:
                manifest["status"] = "final_panel_already_consumed"
                _write(manifest_path, manifest)
                return 3
            final_path = Path("/vol/results") / f"final_{args.tag}.json"
            final_command = remote_command("scripts.evaluate_panel", [
                "--purpose", "final", "--selection", str(selection_path),
                "--out", str(final_path),
            ], gpu=False)
            print(f"[campaign] RUN immutable-final\n  {shlex.join(final_command)}")
            result = subprocess.run(final_command, cwd=REPO)
            manifest["final_evaluation_uses"] = 1
            manifest["final_report"] = str(final_path)
            if result.returncode == 0:
                local_final = config.RESULTS_DIR / f"final_{args.tag}.json"
                fetch = fetch_volume(final_path, local_final)
                if fetch.returncode == 0:
                    final_report = json.loads(local_final.read_text())
                    achieved = final_report["mean_win_rate"]
                    manifest["final_mean_win_rate"] = achieved
                    manifest["final_per_seed"] = [
                        {"seed": item["seed"], "win_rate": item["win_rate"],
                         "ci95": item["win_rate_ci95"]}
                        for item in final_report["per_seed"]]
                    manifest["final_ci95"] = final_report[
                        "pooled_win_rate_ci95"]
                    manifest["status"] = (
                        "success" if achieved >= config.CAMPAIGN[
                            "target_final_win_rate"]
                        else "final_below_target_protocol_complete")
                else:
                    manifest["status"] = "final_report_fetch_failed"
            else:
                manifest["status"] = "final_evaluation_failed"
            _write(manifest_path, manifest)
            # Stop after the one immutable panel. Reading it for reporting is
            # allowed; using it to launch another round is not.
            return result.returncode

    manifest["status"] = (
        "development_threshold_not_met_before_round_limit"
        if args.execute else "dry_run_complete")
    _write(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
