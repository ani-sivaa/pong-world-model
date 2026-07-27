"""Evaluate one frozen policy on a preregistered development or final panel."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import config
from agents.evaluate import eval_policy
from agents.policy import load_policy


def wilson_interval(successes, trials, z=1.959963984540054):
    if trials <= 0:
        return [None, None]
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(
        p * (1 - p) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return [center - margin, center + margin]


def evaluate_panel(policy_path, seeds, episodes, purpose, device):
    policy = load_policy(policy_path, device).eval()
    started = time.time()
    per_seed = []
    for seed in seeds:
        stats, _ = eval_policy(policy, episodes, device, seed=seed, record_first=0)
        stats["seed"] = seed
        stats["win_rate_ci95"] = wilson_interval(stats["wins"], stats["episodes"])
        per_seed.append(stats)
    total_episodes = sum(item["episodes"] for item in per_seed)
    total_wins = sum(item["wins"] for item in per_seed)
    return {
        "schema_version": 1,
        "purpose": purpose,
        "policy": str(policy_path),
        "policy_sha256": hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),
        "seeds": list(seeds),
        "episodes_per_seed": episodes,
        "per_seed": per_seed,
        "mean_win_rate": total_wins / total_episodes,
        "pooled_win_rate_ci95": wilson_interval(total_wins, total_episodes),
        "total_wins": total_wins,
        "total_episodes": total_episodes,
        "wall_seconds": time.time() - started,
        "policy_frozen": True,
        "policy_updates_from_evaluation": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy")
    parser.add_argument("--selection",
                        help="candidate-selection JSON; uses selected_policy")
    parser.add_argument("--purpose", choices=("development", "final"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if bool(args.policy) == bool(args.selection):
        raise ValueError("provide exactly one of --policy or --selection")
    policy = args.policy
    if args.selection:
        selection = json.loads(Path(args.selection).read_text())
        policy = selection.get("selected_policy")
        if not selection.get("eligible") or not policy:
            raise ValueError("selection report contains no eligible candidate")
    expected_seeds = tuple(config.ROUND_TWO[
        "final_evaluation_seeds" if args.purpose == "final"
        else "development_seeds"])
    expected_episodes = int(config.ROUND_TWO[
        "final_episodes_per_seed" if args.purpose == "final"
        else "development_episodes_per_seed"])
    seeds = tuple(args.seeds or expected_seeds)
    episodes = args.episodes or expected_episodes
    if args.purpose == "final" and (
            seeds != expected_seeds or episodes != expected_episodes):
        raise ValueError(
            "final panel seeds and 200 episodes/seed are immutable and preregistered")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable evaluation report: {out}")
    config.seed_everything(0)
    report = evaluate_panel(
        policy, seeds, episodes, args.purpose, config.get_device())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"{args.purpose.upper()}_PANEL_OK "
          f"mean_win_rate={report['mean_win_rate']:.4f} "
          f"ci95={report['pooled_win_rate_ci95']} report={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
