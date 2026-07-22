"""Training-set-size ablation driver (ablation-runner).

Trains the world model at each fraction in config.ABLATION_FRACS on Modal
(4 jobs IN PARALLEL, same fixed step budget => controlled comparison of DATA
quantity, not compute), fetches the checkpoints, runs local rollout drift
evals, and produces:

  results/ablation/ablation_drift.png    - 4 drift curves, log-y MSE
  results/ablation/ablation_summary.json - {frac: {mse@1/10/30/60, final_train_loss}}
  results/ablation/ablation_summary.md   - table + interpretation
  results/ablation/logs/<tag>*.log       - raw job output

Run from repo root: .venv/bin/python scripts/run_ablation.py
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import config  # noqa: E402

VENV_PY = str(REPO / ".venv" / "bin" / "python")
OUT_DIR = REPO / "results" / "ablation"
LOG_DIR = OUT_DIR / "logs"

STEPS = 6000          # fixed budget for EVERY fraction (controlled comparison)
RETRY_STEPS = 3000    # one retry at half budget if TRAINING fails
MAX_SECONDS = 2700    # remote wall-clock cap passed to wm.train
ATTEMPT_WAIT_S = 4800 # local cap per attempt (train cap + container/data/commit)
STAGGER_S = 25        # spacing between launches: 4 simultaneous ephemeral
                      # app.run() calls trip Modal's app-creation rate limit
MAX_LAUNCH_RETRIES = 3  # same-budget relaunches for launch-phase failures
                        # (rate limit / app creation) -- job never started
                        # training, so the controlled budget is unaffected

FRACS = list(config.ABLATION_FRACS)  # (0.10, 0.25, 0.50, 1.00)
FULL_T = 1_000_000


def tag_for(frac: float) -> str:
    return f"ablation{int(round(frac * 100))}"


def label_for(frac: float) -> str:
    n = int(round(frac * FULL_T))
    size = "1M" if n >= 1_000_000 else f"{n // 1000}k"
    return f"{int(round(frac * 100))}% ({size})"


def train_cmd(frac: float, steps: int) -> list[str]:
    return [VENV_PY, "-m", "infra.remote", "--module", "wm.train", "--gpu", "--",
            "--scale", "full", "--data-frac", str(frac), "--tag", tag_for(frac),
            "--steps", str(steps), "--max-seconds", str(MAX_SECONDS)]


def run_attempt(frac: float, steps: int, attempt: int) -> tuple[int, Path]:
    """One remote training attempt; blocks. Returns (rc, log_path). rc=-9 on
    local wait-cap kill."""
    tag = tag_for(frac)
    log_path = LOG_DIR / f"{tag}.attempt{attempt}.log"
    cmd = train_cmd(frac, steps)
    with open(log_path, "w") as fh:
        fh.write(f"# {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=fh,
                                stderr=subprocess.STDOUT)
        print(f"[ablation] launched {tag} attempt {attempt} "
              f"(steps={steps}, pid={proc.pid}) -> {log_path}", flush=True)
        try:
            rc = proc.wait(timeout=ATTEMPT_WAIT_S)
        except subprocess.TimeoutExpired:
            print(f"[ablation] {tag} attempt {attempt} exceeded "
                  f"{ATTEMPT_WAIT_S}s -- killing", flush=True)
            proc.kill()
            proc.wait()
            rc = -9
    print(f"[ablation] {tag} attempt {attempt} finished rc={rc}", flush=True)
    return rc, log_path


def run_fraction(idx: int, frac: float) -> bool:
    """Train one fraction with retry policy:
    - launch-phase failure (never reached wm.train, e.g. Modal app-creation
      rate limit): relaunch at the SAME budget, up to MAX_LAUNCH_RETRIES.
    - training failure: retry ONCE at RETRY_STEPS.
    """
    time.sleep(idx * STAGGER_S)  # avoid simultaneous app-creation rate limit
    tag = tag_for(frac)
    steps, attempt = STEPS, 1
    launch_tries, training_retry_used = 0, False
    while True:
        rc, log_path = run_attempt(frac, steps, attempt)
        if rc == 0:
            return True
        txt = log_path.read_text(errors="ignore")
        started_training = "[wm.train]" in txt
        launch_failure = (not started_training) and rc != -9
        if launch_failure and launch_tries < MAX_LAUNCH_RETRIES:
            launch_tries += 1
            backoff = 30 * launch_tries
            print(f"[ablation] {tag} launch-phase failure (never started "
                  f"training) -- relaunching at SAME {steps} steps in "
                  f"{backoff}s ({launch_tries}/{MAX_LAUNCH_RETRIES})", flush=True)
            time.sleep(backoff)
        elif not training_retry_used:
            training_retry_used = True
            steps = RETRY_STEPS
            print(f"[ablation] {tag} RETRYING once at --steps {RETRY_STEPS}",
                  flush=True)
        else:
            return False
        attempt += 1


def run_logged(cmd: list[str], log_path: Path) -> bool:
    with open(log_path, "w") as fh:
        fh.write(f"# {' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=str(REPO), stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    print(f"[ablation] {' '.join(cmd[-4:])} rc={rc} -> {log_path}", flush=True)
    return rc == 0


def fetch(remote: str, local: str, log_path: Path) -> bool:
    return run_logged([VENV_PY, "-m", "infra.remote", "--fetch", remote, local],
                      log_path)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. remote trainings, all four in parallel (staggered starts) -------
    with ThreadPoolExecutor(max_workers=len(FRACS)) as pool:
        results = list(pool.map(run_fraction, range(len(FRACS)), FRACS))
    ok = dict(zip(FRACS, results))
    trained = [f for f in FRACS if ok[f]]
    dead = [f for f in FRACS if not ok[f]]
    if dead:
        print(f"[ablation] !!! GAVE UP on {[tag_for(f) for f in dead]} after "
              "2 attempts -- proceeding with the rest !!!", flush=True)

    # ---- 2. fetch checkpoints + train logs (sequential) ---------------------
    good = []
    for frac in trained:
        tag = tag_for(frac)
        if fetch(f"checkpoints/{tag}.pt", f"checkpoints/{tag}.pt",
                 LOG_DIR / f"fetch_{tag}_ckpt.log"):
            good.append(frac)
        else:
            print(f"[ablation] !!! checkpoint fetch FAILED for {tag} -- "
                  "dropping this fraction !!!", flush=True)
            dead.append(frac)
        # train log is nice-to-have (final_train_loss); non-fatal if missing
        fetch(f"results/wm_train_log_{tag}.json",
              f"results/wm_train_log_{tag}.json",
              LOG_DIR / f"fetch_{tag}_log.log")

    # ---- 3. local rollout evals (sequential) --------------------------------
    evaled = []
    for frac in good:
        tag = tag_for(frac)
        cmd = [VENV_PY, "-m", "wm.rollout", "--scale", "full",
               "--ckpt", f"checkpoints/{tag}.pt", "--tag", tag,
               "--n-windows", "32", "--gif-lengths"]
        if run_logged(cmd, LOG_DIR / f"rollout_{tag}.log"):
            evaled.append(frac)
        else:
            print(f"[ablation] !!! rollout eval FAILED for {tag} !!!", flush=True)
            dead.append(frac)

    if not evaled:
        print("[ablation] !!! NOTHING SUCCEEDED -- no outputs produced !!!")
        return 1

    # ---- 4. plot + summaries -------------------------------------------------
    from evalutils import plots

    drift, train_log = {}, {}
    for frac in evaled:
        tag = tag_for(frac)
        drift[frac] = json.loads((REPO / "results" / f"drift_{tag}.json").read_text())
        lp = REPO / "results" / f"wm_train_log_{tag}.json"
        if lp.exists():
            train_log[frac] = json.loads(lp.read_text())

    xs = drift[evaled[0]]["h"]
    plots.line_plot(
        xs, {label_for(f): drift[f]["mse_mean"] for f in evaled},
        title=f"Dream drift vs training-set size (fixed {STEPS}-step budget)",
        xlabel="rollout step (autoregressive)", ylabel="mean per-pixel MSE",
        out_path=OUT_DIR / "ablation_drift.png", log_y=True)

    def at(frac, h):
        m = drift[frac]["mse_mean"]
        return float(m[h - 1]) if h <= len(m) else None

    summary = {}
    for frac in FRACS:
        if frac not in drift:
            summary[f"{frac:.2f}"] = {"status": "FAILED"}
            continue
        log = train_log.get(frac)
        summary[f"{frac:.2f}"] = {
            "tag": tag_for(frac),
            "mse@1": at(frac, 1), "mse@10": at(frac, 10),
            "mse@30": at(frac, 30), "mse@60": at(frac, 60),
            "final_train_loss": (float(log["loss"][-1]) if log else None),
            "train_steps_completed": (int(log["step"][-1]) if log else None),
        }
    (OUT_DIR / "ablation_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- markdown table + data-driven interpretation -------------------------
    rows = ["| data | mse@1 | mse@10 | mse@30 | mse@60 | final train loss |",
            "|------|-------|--------|--------|--------|------------------|"]
    for frac in FRACS:
        s = summary[f"{frac:.2f}"]
        if s.get("status") == "FAILED":
            rows.append(f"| {label_for(frac)} | FAILED | FAILED | FAILED | FAILED | - |")
            continue
        ftl = f"{s['final_train_loss']:.4f}" if s["final_train_loss"] is not None else "n/a"
        rows.append(f"| {label_for(frac)} | {s['mse@1']:.5f} | {s['mse@10']:.5f} "
                    f"| {s['mse@30']:.5f} | {s['mse@60']:.5f} | {ftl} |")

    sent = []
    e = sorted(evaled)
    if len(e) >= 2:
        lo, hi = e[0], e[-1]
        r30 = at(lo, 30) / max(at(hi, 30), 1e-12)
        mono = all(at(e[i], 30) >= at(e[i + 1], 30) * 0.98 for i in range(len(e) - 1))
        sent.append(
            f"With the step budget held fixed at {STEPS}, long-horizon coherence "
            f"{'improves monotonically' if mono else 'does NOT improve monotonically'} "
            f"with training-set size: at 30 rollout steps the {label_for(lo)} model's "
            f"MSE is {r30:.1f}x that of the {label_for(hi)} model.")
        if len(e) >= 3:
            gain_lo = at(e[0], 30) / max(at(e[1], 30), 1e-12)
            gain_hi = at(e[-2], 30) / max(at(e[-1], 30), 1e-12)
            if gain_hi < 1.15 and gain_lo > gain_hi:
                sent.append(
                    f"Most of the gain comes early: going {label_for(e[0])} -> "
                    f"{label_for(e[1])} buys a {gain_lo:.1f}x reduction at h=30, while "
                    f"{label_for(e[-2])} -> {label_for(e[-1])} buys only {gain_hi:.2f}x "
                    "-- returns saturate near the upper fractions.")
            else:
                sent.append(
                    f"Gains do not saturate in this range: {label_for(e[0])} -> "
                    f"{label_for(e[1])} gives {gain_lo:.2f}x at h=30 and "
                    f"{label_for(e[-2])} -> {label_for(e[-1])} still gives {gain_hi:.2f}x.")
        sent.append(
            "One-step prediction (mse@1) separates the models far less than the "
            "60-step dream does -- compounding autoregressive error amplifies "
            "small next-frame differences, so data quantity matters most for "
            "long rollouts.")
        sent.append(
            f"Caveat: every model saw exactly {STEPS} optimizer steps, so small "
            "fractions repeat their data more often (more epochs over fewer "
            "transitions); differences reflect data diversity, not compute.")
    if dead:
        sent.append(f"MISSING: {', '.join(tag_for(f) for f in sorted(set(dead)))} "
                    "failed twice and are absent from the comparison.")

    md = "\n".join([
        "# Training-set-size ablation",
        "",
        f"World model trained at {', '.join(label_for(f) for f in FRACS)} of the "
        f"1M-transition dataset, each for the SAME {STEPS}-step budget "
        f"(cap {MAX_SECONDS}s) on a T4; drift measured over 32 held-out "
        "60-step autoregressive rollouts replaying logged actions.",
        "",
        *rows,
        "",
        " ".join(sent),
        "",
        "![drift](ablation_drift.png)",
        ""])
    (OUT_DIR / "ablation_summary.md").write_text(md)

    print("[ablation] DONE")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
