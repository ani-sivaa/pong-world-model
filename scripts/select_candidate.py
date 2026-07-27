"""Select a final candidate using trust and development evidence only."""
import argparse
import json
from pathlib import Path

import config


def strict_trust(report):
    gates = report.get("gates", {})
    return report.get("overall_pass") is True and bool(gates) and all(
        gate.get("available", False) and gate.get("pass") is True
        for gate in gates.values())


def select_candidate(policies, trusts, developments, threshold):
    eligible = []
    records = []
    for index, (policy, trust, development) in enumerate(
            zip(policies, trusts, developments)):
        trusted = strict_trust(trust)
        win_rate = development.get("mean_win_rate")
        passes = trusted and win_rate is not None and win_rate >= threshold
        record = {
            "candidate_index": index,
            "policy": str(policy),
            "strict_trust_pass": trusted,
            "development_mean_win_rate": win_rate,
            "development_ci95": development.get("pooled_win_rate_ci95"),
            "eligible": passes,
            "failed_trust_gates": [
                name for name, gate in trust.get("gates", {}).items()
                if not gate.get("available", False) or gate.get("pass") is not True
            ],
        }
        records.append(record)
        if passes:
            eligible.append(record)
    # Predeclared rule: highest development mean, then lowest candidate index.
    selected = sorted(
        eligible,
        key=lambda item: (-item["development_mean_win_rate"],
                          item["candidate_index"]))
    selected = selected[0] if selected else None
    return {
        "schema_version": 1,
        "selection_uses_final_results": False,
        "development_threshold": threshold,
        "candidates": records,
        "eligible": selected is not None,
        "selected_policy": selected["policy"] if selected else None,
        "selected_candidate_index": (
            selected["candidate_index"] if selected else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", nargs=3, required=True)
    parser.add_argument("--trust", nargs=3, required=True)
    parser.add_argument("--development", nargs=3, required=True)
    parser.add_argument("--threshold", type=float, default=config.ROUND_TWO[
        "development_promotion_win_rate"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = select_candidate(
        args.policies,
        [json.loads(Path(path).read_text()) for path in args.trust],
        [json.loads(Path(path).read_text()) for path in args.development],
        args.threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print("CANDIDATE_" + ("SELECTED" if report["eligible"] else "NONE"), out)


if __name__ == "__main__":
    main()
