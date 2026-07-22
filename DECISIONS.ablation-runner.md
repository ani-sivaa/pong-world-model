# DECISIONS — ablation-runner

## Step-budget choice: fixed 6000 steps for every fraction

- The question the ablation answers is "how does DATA quantity affect rollout
  coherence", so compute must be held constant: every fraction (10/25/50/100%
  of the 1M-transition `data/full` set) trains for exactly `--steps 6000`,
  batch 128, identical seed/arch/hyperparameters. Only `--data-frac` varies.
- 6000 steps is deliberately NOT max quality (the flagship wm_v1 run uses
  18k). 6000 * 128 = 768k sampled windows is enough for the loss to be well
  into its slow tail on this model (per the wm_v1 training curve) while
  keeping 4 parallel T4 jobs inside a 45-min window; the comparison is
  between fractions, not against wm_v1.
- `--max-seconds 2700` guards the wall clock; wm.train checkpoints and exits
  cleanly at the cap, and the summary records `train_steps_completed` from
  the fetched train log so a capped run is visible rather than silent.
- Consequence to keep in mind when reading results: with steps fixed, the 10%
  model does ~10x more epochs over its 100k transitions than the 100% model
  does over 1M — differences measure data DIVERSITY (memorization/overfitting
  vs generalization), not training length.

## Orchestration

- `scripts/run_ablation.py` launches the 4 `infra.remote --module wm.train`
  jobs with `subprocess.Popen` in parallel, STAGGERED 25 s apart (separate
  Modal containers, distinct tags ablation10/25/50/100 — no artifact
  collisions, and no contact with the unrelated wm_v1 run). Output captured
  to `results/ablation/logs/<tag>.attempt<N>.log`.
- Failure policy (two tiers, deliberately distinguished):
  - LAUNCH-phase failure — the job died before `[wm.train]` ever printed
    (e.g. Modal "App creation failed: rate limit exceeded"): relaunch at the
    SAME 6000 steps with backoff, up to 3 times. No training happened, so
    the controlled budget is untouched; burning the half-budget retry on
    this would silently break the comparison.
  - TRAINING failure (crash after training started, or exceeding the 4800 s
    local wait cap): retried ONCE at `--steps 3000`. A fraction that fails
    twice is dropped and flagged loudly in ablation_summary.{json,md} rather
    than blocking the rest.
- Artifact retrieval uses `infra.remote --fetch` exclusively (raw
  `modal volume get` is broken on this network; the wrapper falls back to the
  function-channel chunk reader). Checkpoint fetch failure drops the
  fraction; train-log fetch failure only costs the `final_train_loss` column.
- Rollout evals run locally and sequentially (`wm.rollout --scale full
  --n-windows 32 --gif-lengths` with no lengths => no GIFs), on the identical
  local `data/full` (same seed), horizon 60 from SCALES["full"], so all four
  models are scored on the same 32 held-out windows replaying logged actions.

## Failures / retries observed

- First driver launch died instantly: shell redirect target
  `results/ablation/logs/driver.log` preceded directory creation. Fixed by
  mkdir before relaunch; not a pipeline failure.
- Second driver launch: firing all 4 `app.run()` calls in the same instant
  tripped Modal's app-creation rate limit — ablation25 and ablation50 failed
  in ~20 s with `modal.exception.ServiceError: App creation failed: rate
  limit exceeded` before any training. Killed the driver (so the 3000-step
  retry wouldn't be spent on a launch bug), added the 25 s launch stagger and
  the launch-retry tier described above, and restarted clean. The unrelated
  wm_v1 processes were left untouched.
- (further entries appended after the run if any)

## Interpretation notes

- (filled in after the run — see results/ablation/ablation_summary.md)
