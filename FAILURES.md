# FAILURES.md — errors, fallbacks triggered, and how the run recovered

(empty is the goal; every entry includes traceback, phase, scale reduction or
skip decision, and impact on deliverables)

## [infra self-test — expected failure, ignore] MODAL FAILURE — falling back to local — 2026-07-22T16:15:43+00:00
- module: `infra.does_not_exist` (deliberately bogus; verifies the run_anywhere fallback path)
- argv: `['--scale', 'full']`
- remote error:

```
Traceback (most recent call last):
  File "/root/infra/remote.py", line 68, in _run_module_body
    runpy.run_module(module, run_name="__main__")
  File "<frozen runpy>", line 222, in run_module
  File "<frozen runpy>", line 142, in _get_module_details
ImportError: No module named infra.does_not_exist
```

- action: local retry: `/Users/anisiva/Developer/WorldModel/.venv/bin/python -m infra.does_not_exist --scale local`
- local fallback result: ok=False returncode=1 seconds=0.03

## Modal remote cancellation — iteration-2 chain, stage 3 — 2026-07-22 ~15:10 PT
- `wm.train --v2` (heads fine-tune) died with `modal.exception.RemoteError:
  Function call was cancelled by user or a failure` — Modal-side cancellation
  (no local kill issued; likely worker preemption). Stages 1–2 (mixed collect,
  wm_v3) completed and persisted to the volume before the failure.
- Action: relaunched the chain from stage 3 (heads → dream v2). No scale
  reduction needed.

## Modal GPU scheduling degraded — iteration-2 stages 3-4 moved LOCAL — 2026-07-22 ~15:40 PT
- Three consecutive GPU-job failures (`RemoteError: cancelled`, `ConflictError:
  APP_STATE_STOPPED`), then a GPU hello that hung >7 min while a CPU hello
  returned ok=true in seconds → T4 capacity/scheduling issue on Modal's side,
  not our code (same commands ran fine for hours earlier today).
- Fallback per unattended rules: regenerate mixed data locally (same seed),
  fetch wm_v3.pt from the volume, run heads fine-tune (5k steps) and dream v2
  (1,200 updates, batch 128, ball-guard) on local MPS with wall-clock caps.
  Reduced scale is logged; v1 artifacts untouched.

## Third T4 preemption — dream v2 stopped at update 1,100/2,500 — 2026-07-22 ~16:20 PT
- Heads (wm_v3h) completed (val 0.00020); dream v2 preempted mid-run, last
  checkpoint update 1,100. v1 plateaued by ~1,000 updates, so evaluating the
  1,100-update checkpoint directly instead of fighting Modal's GPU queue.

## Honest campaign local controller lost mid-cycle — 2026-08-03 → 2026-08-10
- Phase: `honest_campaign_v1` execute (`max_rounds=2`), after r0 pretrust FAIL
  and during/after r1 WM training.
- Symptom: fresh Cloud Agent VM had no local controller, no tmux session, no
  campaign lock, empty local `stages` in the tracked manifest, and no
  `campaign_logs/`. Modal `app list` showed zero running apps.
- Impact: local progress bookkeeping was lost; volume retained completed
  r0/r1 collect+WM+pretrust artifacts (both trust FAIL; no policies).
- Recovery: did **not** relaunch a second campaign controller (would redo
  paid stages from empty local stages). Reconstructed manifest stages from
  volume-fetched evidence, wrote RESULTS notebook, marked cycle stopped
  after ≤2 pretrust failures with no promotion/demo export.
- Collision check: only this Cloud Agent was RUNNING on the branch; no
  second controller started.

## Modal spend-limit hard stop — honest_campaign_v2 round-2 policy-0 — 2026-08-13
- Phase: `honest_campaign_v2` execute, after r2 pretrust PASS, during `round-2:policy-0`
  dream PPO (seed 42).
- Symptom: first attempt reached **update 180/2500** (~1795s) then Modal raised
  `ResourceExhaustedError: Workspace ... has exceeded its spend limit`. Retries
  hit `ConflictError: workspace ... is disabled`. Billing snapshot: metered ≈
  $37.96 / credits applied ≈ $30 / billed ≈ $6.59.
- Impact: controller stopped; no further dream-policy seeds; no development
  panel for round-2; no promotion; no demo export; final panel unused.
- Action: recorded `budget_or_auth_exhausted` in campaign manifest; widened
  `STOP_ERROR` regex to catch spend-limit / workspace-disabled so future runs
  do not burn retries. Did **not** relaunch paid jobs.
