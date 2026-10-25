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

## Dream wall-clock cap — honest_campaign_v2 r2 policy-2 — 2026-10-13
- Phase: `agents.train_dream` seed 2718 under `CAPS.dream=14400`.
- Symptom: `[dream] WALL-CLOCK CAP at update 1465` then `DREAM_OK` with
  checkpoint `/vol/checkpoints/honest_campaign_v2_r2_ppo_seed2718.pt`.
  Not a Modal spend-limit failure (billing remained billed $0).
- Impact: policy-2 entered policy-trust/development with a partial update
  count (1465/2500). Trust still **PASS**; development mean win rate 0%.
- Action: accepted as protocol-compliant wall stop; volume checkpoint kept
  for a future `--init-from` resume. No demo export; final panel unused.

## Round-3 pretrust ballless fail — honest_campaign_v2 — 2026-10-14
- Phase: transfer-repair flywheel after r2 0% development collapse.
- Collect+3×WM completed on mix rare=.35; latent utilization **passed**
  (4.11e-5). `ballless_positive_rate` **failed** at 0.122 > 0.05.
- Action: did **not** train dream policies. Strengthened ballless training
  penalty (coef 6.0, tau 0.35) and queued round 4 under calibration mix
  (rare=.40). No demo export; final panel unused.

## WM step-resume CosineAnnealingLR initial_lr — 2026-10-15
- Phase: `honest_campaign_v2` round-4:wm-0 resume via `--init-from` after
  adding saved-step continuation.
- Symptom: Modal GPU job failed immediately with
  `KeyError: param 'initial_lr' is not specified in param_groups[0] when
  resuming scheduler with last_epoch >= 0`.
- Action: moved the already-complete early-return before optimizer creation;
  anneal CosineAnnealingLR over the *remaining* step budget instead of
  `last_epoch` without optimizer state. Cleared failed `round-4:wm-0` stage
  and relaunched. No collect reburn; no demo export.

## Round-4 pretrust ballless near-miss — honest_campaign_v2 — 2026-10-15
- Phase: calibration-mix r4 after strengthening ballless_reward_coef to 6.0.
- Collect recovered; wm0/wm1 no-op; wm2 resumed 11000→18000.
- Latent utilization **passed** (5.84e-5). `ballless_positive_rate` **failed**
  at **0.071** (4/56 ballless frames) — improved from r3's 0.122 but still
  above the 0.05 gate.
- Action: did **not** train dream policies. Planned same-data fine-tune with
  stronger ballless hinge (coef 12, tau 0.20, threshold-aligned relu(r-0.05)).
  No demo export; final panel unused.

## Modal spend limit — r4 ballless fine-tune wm-1 — 2026-10-15
- Phase: same-data ballless fine-tune after r4 pretrust near-miss (0.071).
- wm-0 completed 18000→21000 under coef=12/tau=0.20/hinge=0.05 (val 0.00595).
- wm-1 reached step **20000**/21000 then Modal returned
  `ConflictError: workspace ... is disabled` /
  `Workspace ... has exceeded its spend limit`. GPU smoke afterward fails the
  same spend-limit gate. Billing snapshot: metered ≈ $31.94 / billed ≈ $0.05.
- Impact: wm-2 never fine-tuned (still step 18000, coef=6). Ensemble is
  intentionally **not** re-pretrusted while members disagree on the ballless
  hinge. No dream PPO; no demo export; final panel unused.
- Action: recorded hard stop in campaign manifest
  (`budget_or_auth_exhausted`); stopped paid launches for this cycle.

## Modal spend limit still blocks daily resume — 2026-10-16
- Phase: daily honest research cron on
  `cursor/honest-research-cycle-c8aa` (continued from `3c24@f318a07`).
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; `modal token set --verify` **OK**;
  profile `default`; billing summary readable (metered ≈ $32.07 / credits
  applied ≈ $30 / billed ≈ $0.02). Env balance $25 is start telemetry only.
- Symptom: `modal run scripts/smoke_modal.py` and CPU
  `infra.remote --module infra.hello` both raise
  `ResourceExhaustedError: Workspace ... has exceeded its spend limit`.
  Volume `ls`/`get` still succeed.
- Volume audit (fetched ckpt metadata only):
  - `r4_wm0.pt` step **21000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm1.pt` step **20000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm2.pt` step **18000**, coef=6 / tau=0.35 (not fine-tuned)
- Impact: cannot finish ballless FT, cannot re-pretrust, cannot dream-train.
  No collect→WM burn attempted (science rule: repair first).
- Action: recorded hard stop + resume attempt in campaign manifest; stopped
  paid launches. Local unit tests still pass (27). Evidence:
  `results/campaign_logs/modal_smoke_2026-10-16.log`,
  `volume_ckpt_audit_2026-10-16.log`, `unit_tests_2026-10-16.log`.

## Modal spend limit still blocks daily resume — 2026-10-17
- Phase: daily honest research cron on
  `cursor/honest-research-cycle-f4b7` (continued from latest
  `c8aa`/`c57f` campaign tip `bd4121b`).
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; `modal token set --verify` **OK**;
  profile `ani-sivaa`; billing summary readable (metered ≈ $32.22 / credits
  applied ≈ $30 / billed ≈ $0.02). Env balance $25 is start telemetry only.
- Symptom: `modal run scripts/smoke_modal.py` and CPU
  `infra.remote --module infra.hello` both raise
  `ResourceExhaustedError: Workspace ... has exceeded its spend limit`.
  Volume `ls`/`get` still succeed.
- Volume audit (fetched ckpt metadata only):
  - `r4_wm0.pt` step **21000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm1.pt` step **20000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm2.pt` step **18000**, coef=6 / tau=0.35 (not fine-tuned)
- Impact: cannot finish ballless FT, cannot re-pretrust, cannot dream-train.
  No collect→WM burn attempted (science rule: repair first).
- Action: recorded hard stop + resume attempt in campaign manifest; stopped
  paid launches. Local unit tests still pass (33). Evidence:
  `results/campaign_logs/modal_smoke_2026-10-17.log`,
  `volume_ckpt_audit_2026-10-17.log`, `unit_tests_2026-10-17.log`.

## Modal spend limit still blocks daily resume — 2026-10-18
- Phase: daily honest research cron on
  `cursor/honest-research-cycle-9cb9` (continued from latest
  `f4b7`/`c57f` campaign tip `a7e205f`).
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; `modal token set --verify` **OK**;
  profile `ani-sivaa`; billing summary readable (metered ≈ $32.38 / credits
  applied ≈ $30 / billed ≈ $0.02). Env balance $25 is start telemetry only.
- Symptom: `modal run scripts/smoke_modal.py` and CPU
  `infra.remote --module infra.hello` both raise
  `ResourceExhaustedError: Workspace ... has exceeded its spend limit`.
  Volume `ls`/`get` still succeed.
- Volume audit (fetched ckpt metadata only):
  - `r4_wm0.pt` step **21000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm1.pt` step **20000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm2.pt` step **18000**, coef=6 / tau=0.35 (not fine-tuned)
- Impact: cannot finish ballless FT, cannot re-pretrust, cannot dream-train.
  No collect→WM burn attempted (science rule: repair first; binding FT still
  unfinished).
- Action: recorded hard stop + resume attempt in campaign manifest; stopped
  paid launches. Local unit tests still pass (33). Evidence:
  `results/campaign_logs/modal_smoke_2026-10-18.log`,
  `volume_ckpt_audit_2026-10-18.log`, `unit_tests_2026-10-18.log`.

## Modal spend limit still blocks daily resume — 2026-10-19
- Phase: daily honest research cron on
  `cursor/honest-research-cycle-c155` (continued from latest
  `9cb9`/`c57f` campaign tip `a853efa`).
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; `modal token set --verify` **OK**;
  profile `ani-sivaa`; billing summary readable (metered ≈ $32.54 / credits
  applied ≈ $30 / billed ≈ $0.02). Env balance $25 is start telemetry only.
- Symptom: `infra.remote` GPU smoke and CPU `infra.hello` both raise
  `ResourceExhaustedError: Workspace ... has exceeded its spend limit`.
  Volume `ls`/`get` still succeed.
- Volume audit (fetched ckpt metadata only):
  - `r4_wm0.pt` step **21000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm1.pt` step **20000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm2.pt` step **18000**, coef=6 / tau=0.35 (not fine-tuned)
- Impact: cannot finish ballless FT, cannot re-pretrust, cannot dream-train.
  No collect→WM burn attempted (science rule: repair first; binding FT still
  unfinished).
- Action: recorded hard stop + resume attempt in campaign manifest; stopped
  paid launches. Local unit tests still pass (33). Evidence:
  `results/campaign_logs/modal_smoke_2026-10-19.log`,
  `volume_ckpt_audit_2026-10-19.log`, `unit_tests_2026-10-19.log`.

## Modal spend limit still blocks daily resume — 2026-10-20
- Phase: daily honest research cron on
  `cursor/honest-research-loop-8058` (continued from latest
  `c155`/`c57f` campaign tip `5530748`).
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; `modal token set --verify` **OK**;
  profile `ani-sivaa`; billing summary readable (metered ≈ $32.69 / credits
  applied ≈ $30 / billed ≈ $0.02). Env balance $25 is start telemetry only.
- Symptom: `infra.remote` GPU smoke (`scripts.smoke_modal`) and CPU
  `infra.hello` both raise
  `ResourceExhaustedError: Workspace ... has exceeded its spend limit`.
  Volume `ls`/`get` still succeed.
- Volume audit (fetched ckpt metadata only):
  - `r4_wm0.pt` step **21000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm1.pt` step **20000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm2.pt` step **18000**, coef=6 / tau=0.35 (not fine-tuned)
- Impact: cannot finish ballless FT, cannot re-pretrust, cannot dream-train.
  No collect→WM burn attempted (science rule: repair first; binding FT still
  unfinished).
- Action: recorded hard stop + resume attempt in campaign manifest; stopped
  paid launches. Local unit tests still pass (33). Evidence:
  `results/campaign_logs/modal_smoke_2026-10-20.log`,
  `volume_ckpt_audit_2026-10-20.log`, `unit_tests_2026-10-20.log`.

## Modal spend limit still blocks daily resume — 2026-10-21
- Phase: daily honest research cron on
  `cursor/pong-world-model-research-4679` (continued from latest
  `honest-research-loop-8058` tip `5237f1e` / base `c57f`).
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; `modal token set --verify` **OK**;
  profile `ani-sivaa`; billing summary readable (metered ≈ $32.69 / credits
  applied ≈ $30 / billed ≈ $0.02). Env balance $25 is start telemetry only.
- Symptom: `infra.remote` GPU smoke (`scripts.smoke_modal`) and CPU
  remote hello both raise
  `ResourceExhaustedError: Workspace ... has exceeded its spend limit`.
  Volume `ls`/`get` still succeed.
- Volume audit (fetched ckpt metadata only):
  - `r4_wm0.pt` step **21000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm1.pt` step **20000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm2.pt` step **18000**, coef=6 / tau=0.35 (not fine-tuned)
- Impact: cannot finish ballless FT, cannot re-pretrust, cannot dream-train.
  No collect→WM burn attempted (science rule: repair first; binding FT still
  unfinished).
- Action: recorded hard stop + resume attempt in campaign manifest; stopped
  paid launches. Local unit tests still pass (33). Evidence:
  `results/campaign_logs/modal_smoke_2026-10-21.log`,
  `volume_ckpt_audit_2026-10-21.log`, `unit_tests_2026-10-21.log`.

## Modal spend limit still blocks daily resume — 2026-10-22
- Phase: daily honest research cron on
  `cursor/pong-world-model-research-e88d` (continued from latest
  `pong-world-model-research-4679` tip `5ceaa82` / base `c57f`).
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; `modal token set --verify` **OK**;
  profile `ani-sivaa`; billing summary readable (metered ≈ $32.69 / credits
  applied ≈ $30 / billed ≈ $0.02). Env balance $25 is start telemetry only.
- Symptom: `infra.remote` GPU smoke (`scripts.smoke_modal`) and CPU
  remote hello both raise
  `ResourceExhaustedError: Workspace ... has exceeded its spend limit`.
  Volume `ls`/`get` still succeed.
- Volume audit (fetched ckpt metadata only):
  - `r4_wm0.pt` step **21000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm1.pt` step **20000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm2.pt` step **18000**, coef=6 / tau=0.35 (not fine-tuned)
- Impact: cannot finish ballless FT, cannot re-pretrust, cannot dream-train.
  No collect→WM burn attempted (science rule: repair first; binding FT still
  unfinished).
- Action: recorded hard stop + resume attempt in campaign manifest; stopped
  paid launches. Local unit tests still pass (33). Evidence:
  `results/campaign_logs/modal_smoke_2026-10-22.log`,
  `volume_ckpt_audit_2026-10-22.log`, `unit_tests_2026-10-22.log`.

## Modal spend limit still blocks daily resume — 2026-10-23
- Phase: daily honest research cron on
  `cursor/pong-world-model-research-1033` (continued from latest
  `pong-world-model-research-e88d` tip `6b6c10a` / base `c57f`).
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; `modal token set --verify` **OK**;
  profile `ani-sivaa`; billing summary readable (metered ≈ $32.69 / credits
  applied ≈ $30 / billed ≈ $0.02). Env balance $25 is start telemetry only.
- Symptom: `infra.remote` GPU smoke (`scripts.smoke_modal`) and CPU
  remote hello both raise
  `ResourceExhaustedError: Workspace ... has exceeded its spend limit`.
  Volume `ls`/`get` still succeed.
- Volume audit (fetched ckpt metadata only):
  - `r4_wm0.pt` step **21000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm1.pt` step **20000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm2.pt` step **18000**, coef=6 / tau=0.35 (not fine-tuned)
- Impact: cannot finish ballless FT, cannot re-pretrust, cannot dream-train.
  No collect→WM burn attempted (science rule: repair first; binding FT still
  unfinished).
- Action: recorded hard stop + resume attempt in campaign manifest; stopped
  paid launches. Local unit tests still pass (33). Evidence:
  `results/campaign_logs/modal_smoke_2026-10-23.log`,
  `volume_ckpt_audit_2026-10-23.log`, `unit_tests_2026-10-23.log`.

## Modal spend limit still blocks daily resume — 2026-10-24
- Phase: daily honest research cron on
  `cursor/honest-research-cycle-5100` (continued from latest
  `pong-world-model-research-1033` tip `be81600` / base `c57f`).
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; `modal token set --verify` **OK**;
  profile `ani-sivaa`; billing summary readable (metered ≈ $32.69 / credits
  applied ≈ $30 / billed ≈ $0.02). Env balance $25 is start telemetry only.
- Symptom: `infra.remote` GPU smoke (`scripts.smoke_modal`) and CPU
  remote hello both raise
  `ResourceExhaustedError: Workspace ... has exceeded its spend limit`.
  Volume `ls`/`get` still succeed.
- Volume audit (fetched ckpt metadata only):
  - `r4_wm0.pt` step **21000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm1.pt` step **20000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm2.pt` step **18000**, coef=6 / tau=0.35 (not fine-tuned)
- Impact: cannot finish ballless FT, cannot re-pretrust, cannot dream-train.
  No collect→WM burn attempted (science rule: repair first; binding FT still
  unfinished).
- Action: recorded hard stop + resume attempt in campaign manifest; stopped
  paid launches. Local unit tests still pass (33). Evidence:
  `results/campaign_logs/modal_smoke_2026-10-24.log`,
  `volume_ckpt_audit_2026-10-24.log`, `unit_tests_2026-10-24.log`.

## Modal spend limit still blocks daily resume — 2026-10-25
- Phase: daily honest research cron on
  `cursor/honest-research-cycle-be10` (continued from latest
  `honest-research-cycle-5100` tip `f5bf4a7` / base `c57f`).
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; `modal token set --verify` **OK**;
  profile `ani-sivaa`; billing summary readable (metered ≈ $32.69 / credits
  applied ≈ $30 / billed ≈ $0.02). Env balance $25 is start telemetry only.
- Symptom: `infra.remote` GPU smoke (`scripts.smoke_modal`) and CPU
  remote hello both raise
  `ResourceExhaustedError: Workspace ... has exceeded its spend limit`.
  Volume `ls`/`get` still succeed.
- Volume audit (fetched ckpt metadata only):
  - `r4_wm0.pt` step **21000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm1.pt` step **20000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm2.pt` step **18000**, coef=6 / tau=0.35 (not fine-tuned)
- Impact: cannot finish ballless FT, cannot re-pretrust, cannot dream-train.
  No collect→WM burn attempted (science rule: repair first; binding FT still
  unfinished).
- Action: recorded hard stop + resume attempt in campaign manifest; stopped
  paid launches. Local unit tests still pass (33). Evidence:
  `results/campaign_logs/modal_smoke_2026-10-25.log`,
  `volume_ckpt_audit_2026-10-25.log`, `unit_tests_2026-10-25.log`.
