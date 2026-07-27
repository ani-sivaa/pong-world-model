"""Aggregate round-two evidence without exporting an untrusted policy."""
import argparse
import json
from pathlib import Path

import config


def aggregate_reports(trust_reports, transfer_reports, minimum_win_rate,
                      target_win_rate, max_reward_gap):
    trust_ok = []
    gaps = []
    wins = []
    reasons = []
    for index, (trust, transfer) in enumerate(
            zip(trust_reports, transfer_reports)):
        gates = trust.get("gates", {})
        strict_gate_pass = bool(gates) and all(
            gate.get("available", False) and gate.get("pass") is True
            for gate in gates.values())
        trust_ok.append(strict_gate_pass and trust.get("overall_pass") is True)
        gap_metric = trust.get("measurements", {}).get(
            "current_policy_lockstep", {}).get("dream_real_reward_gap", {})
        gap = gap_metric.get("value") if gap_metric.get("available") else None
        gaps.append(gap)
        policy_stats = transfer.get("dream", {})
        win = policy_stats.get("win_rate")
        wins.append(win)
        if not trust_ok[-1]:
            reasons.append(f"seed {index}: one or more trust gates failed/unavailable")
        if gap is None or gap > max_reward_gap:
            reasons.append(f"seed {index}: reward gap regressed")
        if win is None or win <= minimum_win_rate:
            reasons.append(
                f"seed {index}: win rate did not exceed {minimum_win_rate:.1%}")
    accepted = not reasons and len(wins) == 3
    mean_win = sum(wins) / len(wins) if wins and all(
        value is not None for value in wins) else None
    return {
        "schema_version": 1,
        "accepted": accepted,
        "export_policy": None,
        "policy_seeds": len(wins),
        "all_trust_gates_pass": bool(trust_ok) and all(trust_ok),
        "no_reward_gap_regression": bool(gaps) and all(
            value is not None and value <= max_reward_gap for value in gaps),
        "all_seed_win_rates_above_minimum": bool(wins) and all(
            value is not None and value > minimum_win_rate for value in wins),
        "target_met": mean_win is not None and mean_win > target_win_rate,
        "win_rates": wins,
        "mean_win_rate": mean_win,
        "reward_gaps": gaps,
        "thresholds": {
            "minimum_win_rate_exclusive": minimum_win_rate,
            "target_mean_win_rate_exclusive": target_win_rate,
            "max_reward_gap": max_reward_gap,
        },
        "reasons": reasons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust", nargs=3, required=True)
    parser.add_argument("--transfer", nargs=3, required=True)
    parser.add_argument("--minimum-win-rate", type=float,
                        default=config.ROUND_TWO["minimum_win_rate"])
    parser.add_argument("--target-win-rate", type=float,
                        default=config.ROUND_TWO["target_win_rate"])
    parser.add_argument("--max-reward-gap", type=float,
                        default=config.ROUND_TWO["max_reward_gap"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    report = aggregate_reports(
        [json.loads(Path(path).read_text()) for path in args.trust],
        [json.loads(Path(path).read_text()) for path in args.transfer],
        args.minimum_win_rate, args.target_win_rate, args.max_reward_gap)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print("ROUND_TWO_PROMOTION_" + ("PASS" if report["accepted"] else "REJECT"),
          str(out))
    return 1 if args.enforce and not report["accepted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
