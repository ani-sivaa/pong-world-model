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

## Modal spend limit still blocks daily resume — 2026-10-26
- Phase: daily honest research cron on
  `cursor/honest-research-cycle-929d` (continued from latest
  `honest-research-cycle-be10` tip `f39f0bd` / base `c57f`).
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
  `results/campaign_logs/modal_smoke_2026-10-26.log`,
  `volume_ckpt_audit_2026-10-26.log`, `unit_tests_2026-10-26.log`.

## Modal spend limit still blocks daily resume — 2026-10-27
- Phase: daily honest research cron on
  `cursor/honest-research-loop-760a` (continued from latest
  `honest-research-cycle-929d` tip `a28db36` / base `c57f`).
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
  `results/campaign_logs/modal_smoke_2026-10-27.log`,
  `volume_ckpt_audit_2026-10-27.log`, `unit_tests_2026-10-27.log`.

## Modal spend limit still blocks daily resume — 2026-10-28
- Phase: daily honest research cron on
  `cursor/honest-research-cycle-0e13` (continued from latest
  `honest-research-loop-760a` tip `7365e82` / base `c57f`).
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
  `results/campaign_logs/modal_smoke_2026-10-28.log`,
  `volume_ckpt_audit_2026-10-28.log`, `unit_tests_2026-10-28.log`.

## Modal spend limit still blocks daily resume — 2026-10-29
- Phase: daily honest research cron on
  `cursor/honest-research-cycle-d71b` (continued from latest
  `honest-research-cycle-0e13` tip `2100f60` / base `c57f`).
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
  `results/campaign_logs/modal_smoke_2026-10-29.log`,
  `volume_ckpt_audit_2026-10-29.log`, `unit_tests_2026-10-29.log`.

## Modal spend limit still blocks daily resume — 2026-10-30
- Phase: daily honest research cron on
  `cursor/world-model-research-loop-0929` (continued from latest
  `honest-research-cycle-d71b` tip `f36f489` / base `c57f`).
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
  `results/campaign_logs/modal_smoke_2026-10-30.log`,
  `volume_ckpt_audit_2026-10-30.log`, `unit_tests_2026-10-30.log`.

## Modal spend limit still blocks daily resume — 2026-10-31
- Phase: daily honest research cron on
  `cursor/world-model-research-loop-ef61` (continued from latest
  `world-model-research-loop-0929` tip `1ed8413` / base `c57f`).
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
  `results/campaign_logs/modal_smoke_2026-10-31.log`,
  `volume_ckpt_audit_2026-10-31.log`, `unit_tests_2026-10-31.log`.

## Modal spend limit CLEARED — resume r4 ballless fine-tune — 2026-11-01
- Auth: token verify OK; profile `ani-sivaa`. Env credit telemetry $25 (start only).
- Billing period appears reset: `modal billing summary` metered ≈ $0 at probe time
  (prior cycle metered ≈ $32.69 / credits ≈ $30 / billed ≈ $0.02).
- GPU smoke **PASS** (T4 hello 3.06s); CPU remote hello **PASS** (3.04s).
- Volume audit unchanged pre-resume: wm0@21000 coef12; wm1@20000 coef12;
  wm2@18000 coef6/tau0.35 (not fine-tuned).
- Action: resumed sequential ballless FT under coef=12 / tau=0.20 / hinge=0.05
  on same `/vol/data/honest_campaign_v2_r4` (no new collect→WM burn):
  - wm1: 20000→21000 (seed 314)
  - wm2: 18000→21000 (seed 2718)
  - wm0 already complete; re-pretrust gated until ensemble matches.
- Logs: `results/campaign_logs/r4_ballless_finetune_2026-11-01.log`,
  `round-4_wm-*_ballless_ft_2026-11-01.log`, progress sidecar.

- **Update 2026-11-01 wm-1 complete:** step 20000→**21000**, final val **0.00764**,
  wall ≈ 527s (`WM_TRAIN_OK`). Checkpoint written in-place to
  `/vol/checkpoints/honest_campaign_v2_r4_wm1.pt` under coef=12 / tau=0.20 /
  hinge=0.05. wm-2 fine-tune launching next (18000→21000).

- **Update 2026-11-01 wm-2 complete:** step 18000→**21000** (one mid-run resume
  from step 19000 after Modal retry), final val **0.00925**, wall ≈ 1073s
  (`WM_TRAIN_OK`). In-place ckpt under coef=12 / tau=0.20 / hinge=0.05.
- **Ensemble now matched** at step 21000 / coef=12 for wm0/wm1/wm2. Re-pretrust
  launched next; still no dream PPO until pretrust PASS. Billing mid-cycle
  metered ≈ $0.77 / billed $0 (credits applied).

## r4 ballless-FT re-pretrust FAIL — true-target hinge repair — 2026-11-01
- Matched ensemble re-pretrust: util **PASS** 5.19e-5; ballless **FAIL**
  **0.111** (1 violation / 9 ballless frames). Prior r4 pretrust was 0.071
  (4/56); FT reduced ballless mass but the small denominator fails the rate.
- Predicted-ballless&positive training signal was ~0 during coef=12 FT.
- Concrete repair (no new collect→WM): supervise TRUE ballless targets in
  `wm.train` hinge via `hard_ballless_mask` (pred OR true). Then FT
  21000→23000 under same coef=12 / tau=0.20 / hinge=0.05 and re-pretrust.
- Dream PPO still gated. Demo export unchanged.

- **Update 2026-11-01 true-target FT complete:** parallel wm0/1/2 21000→**23000**
  under coef=12 / tau=0.20 / hinge=0.05 with pred∨true ballless mask.
  Final val: wm0 **0.00740**, wm1 **0.01006**, wm2 **0.00897**. Billing ≈ $1.72
  metered / $0 billed. Re-pretrust launching next.

- **Update 2026-11-01 true-target re-pretrust PASS:** overall **True**.
  util **PASS** 3.61e-5; ballless **PASS** **0.0476** (1 violation / 21
  ballless frames). Dream PPO unlocked with warm-start from r1 seed314.

- **Update 2026-11-01 dream PPO launched:** three parallel imagined PPO runs
  (seeds 42/314/2718), each `--init-from` r1 seed314 with `--warm-start`,
  against the matched r4 true-ballless WM ensemble. Policy gradients remain
  inside the WM only. Logs:
  `results/campaign_logs/round-4_true_ballless_ppo_seed*_2026-11-01.log`.


## Dream PPO mid-run resume — honest_campaign_v2 r4 true-ballless — 2026-11-02
- Phase: continue Nov-1 warm-start dream PPO after pretrust PASS (ballless 0.0476).
- Auth/GPU/CPU probes green; spend limit cleared this billing window.
- Volume audit: policies incomplete at steps seed42@500, seed314@700,
  seed2718@1462 of 2500 (WM ensemble matched @23000 / coef=12).
- Action: resumed each seed with plain `--init-from` (no `--warm-start`) so
  update counters continue. Policy gradients remain inside the frozen WM.
- Next: policy-trust + development panel only after checkpoints reach budget
  or honest wall-stop; no demo export until development promotion.


## Dream wall-clock cap — r4 true-ballless resume — 2026-11-02
- Phase: first resume of incomplete Nov-1 dream PPO seeds.
- seed42: WALL at update **1960**/2500 (`DREAM_OK`, wall 14410s).
- seed314: WALL at update **2143**/2500 (`DREAM_OK`, wall 14411s).
- seed2718: completed **2500**/2500 earlier (no wall).
- Action: second `--init-from` resume (no warm-start) for seeds 42/314 to
  finish the preregistered 2500-update budget before policy-trust/development.

## Modal spend limit — r4 true-ballless ext5k dream PPO — 2026-11-03
- Phase: daily honest research cron on
  `cursor/pong-world-model-research-5dce` (FF from `trust-2112@3e15e94`).
- Auth gate: credentials present; `modal token set --verify` **OK**; GPU+CPU
  smoke **PASS** at start. Pretrust still **PASS** on r4 true-ballless WM
  (util 3.61e-5; ballless 0.0476). Binding gates clear — no collect→WM reburn.
- Action attempted: continue dream PPO 2500→5000 under frozen r4 WM for seeds
  42/314/2718 (policy gradients inside WM only; ball-guard on).
- Progress before stop: all three reached ~update **2660**/5000; volume
  periodic checkpoints at step **2600**
  (`honest_campaign_v2_r4_true_ballless_ext5k_ppo_seed{42,314,2718}.pt`).
- Symptom: `ConflictError: workspace ... is disabled` mid-job; subsequent
  `modal run` returns `Workspace ... has exceeded its spend limit`.
- Billing end: metered ≈ **$30.07** / credits applied ≈ **$30** / billed
  ≈ **$0.07**. Env `MODAL_CREDIT_BALANCE_USD=25` remains start telemetry.
- Impact: cannot finish 5000-update extension; cannot re-run policy-trust or
  development for the incomplete ext5k policies. Prior completed 2500-update
  development leader (**19%** seed314) retained. No demo export; final panel
  unused.
- Action: hard-stop recorded; paid launches stopped for this cycle. Resume
  next when spend limit clears: `--init-from` ext5k@2600 → 5000, then
  policy-trust + development.

## Modal spend limit — daily resume blocked before ext5k finish — 2026-11-04
- Phase: daily honest research cron on
  `cursor/pong-world-model-research-2207` (FF from latest campaign tip
  `cursor/pong-world-model-research-5dce@04eadb4`).
- Auth gate: credentials present; `modal token set --verify` **OK**; profile
  `ani-sivaa`. Env `MODAL_CREDIT_BALANCE_USD=25` is start telemetry only.
- GPU smoke **FAIL** spend limit; CPU remote hello **FAIL** spend limit.
  Volume reads **OK** (direct `modal.Volume.read_file`).
- Billing: metered ≈ **$30.23** / credits applied ≈ **$30** / billed ≈ **$0.07**.
- Binding pretrust still held from prior true-ballless repair (util PASS;
  ballless PASS 0.0476) — **no collect→WM reburn** attempted.
- Planned lever (blocked): resume incomplete ext5k dream PPO 2600→5000 under
  frozen r4 WM for seeds 42/314/2718 (`--init-from`, no warm-start).
- Volume audit confirms ext5k seeds still at step **2600**/5000; completed
  true-ballless policies remain at **2500**; WM ensemble @**23000**/coef=12.
- Impact: cannot advance dream training, policy-trust, or development panel.
  Prior development leader (**19%** seed314) retained vs 12% dream-v2 control.
  Not promoted; final panel unused; demo unchanged; prior 17.5% trust-failed
  stochastic policy not exported.
- Action: hard-stop recorded; paid launches not started. Resume next when
  spend limit clears.

## Modal spend limit — daily resume blocked before ext5k finish — 2026-11-05
- Phase: daily honest research cron on
  `cursor/world-model-research-loop-5eda` (FF from latest campaign tip
  `cursor/pong-world-model-research-2207@9a21696`; base
  `cursor/honest-research-cycle-c57f`).
- Auth gate: credentials present; `modal token set --verify` **OK**; profile
  `ani-sivaa`. Env `MODAL_CREDIT_BALANCE_USD=25` is start telemetry only.
- GPU smoke **FAIL** spend limit; CPU remote hello **FAIL** spend limit.
  Volume reads **OK** (direct `modal.Volume.read_file`).
- Billing: metered ≈ **$30.40** / credits applied ≈ **$30** / billed ≈ **$0.07**.
- Binding pretrust still held from prior true-ballless repair (util PASS
  3.61e-5; ballless PASS 0.0476) — **no collect→WM reburn** attempted.
- Planned lever (blocked): resume incomplete ext5k dream PPO 2600→5000 under
  frozen r4 WM for seeds 42/314/2718 (`--init-from`, no warm-start).
- Volume audit confirms ext5k seeds still at step **2600**/5000; completed
  true-ballless policies remain at **2500**; WM ensemble @**23000**/coef=12.
- Impact: cannot advance dream training, policy-trust, or development panel.
  Prior development leader (**19%** seed314) retained vs 12% dream-v2 control.
  Not promoted; final panel unused; demo unchanged; prior 17.5% trust-failed
  stochastic policy not exported.
- Action: hard-stop recorded; paid launches not started. Resume next when
  spend limit clears.

## Modal spend limit — daily resume blocked before ext5k finish — 2026-11-06
- Phase: daily honest research cron on
  `cursor/world-model-research-loop-68a1` (FF from latest campaign tip
  `cursor/world-model-research-loop-5eda@95275d4`; base
  `cursor/honest-research-cycle-c57f`).
- Auth gate: credentials present; `modal token set --verify` **OK**; profile
  `ani-sivaa`. Env `MODAL_CREDIT_BALANCE_USD=25` is start telemetry only.
- GPU smoke **FAIL** spend limit; CPU remote hello **FAIL** spend limit.
  Volume reads **OK** (direct `modal.Volume.read_file`).
- Billing: metered ≈ **$30.56** / credits applied ≈ **$30** / billed ≈ **$0.07**.
- Binding pretrust still held from prior true-ballless repair (util PASS
  3.61e-5; ballless PASS 0.0476) — **no collect→WM reburn** attempted.
- Planned lever (blocked): resume incomplete ext5k dream PPO 2600→5000 under
  frozen r4 WM for seeds 42/314/2718 (`--init-from`, no warm-start).
- Volume audit confirms ext5k seeds still at step **2600**/5000; completed
  true-ballless policies remain at **2500**; WM ensemble @**23000**/coef=12.
- Impact: cannot advance dream training, policy-trust, or development panel.
  Prior development leader (**19%** seed314) retained vs 12% dream-v2 control.
  Not promoted; final panel unused; demo unchanged; prior 17.5% trust-failed
  stochastic policy not exported.
- Action: hard-stop recorded; paid launches not started. Resume next when
  spend limit clears.

## Modal spend limit — daily resume blocked before ext5k finish — 2026-11-07
- Phase: daily honest research cron on
  `cursor/world-model-research-loop-82da` (FF from latest campaign tip
  `cursor/world-model-research-loop-68a1@59d68d5`; base
  `cursor/honest-research-cycle-c57f`).
- Auth gate: credentials present; `modal token set --verify` **OK**; profile
  `ani-sivaa`. Env `MODAL_CREDIT_BALANCE_USD=25` is start telemetry only.
- GPU smoke **FAIL** spend limit; CPU remote hello **FAIL** spend limit.
  Volume reads **OK** (direct `modal.Volume.read_file`).
- Billing: metered ≈ **$30.56** / credits applied ≈ **$30** / billed ≈ **$0.07**.
- Binding pretrust still held from prior true-ballless repair (util PASS
  3.61e-5; ballless PASS 0.0476) — **no collect→WM reburn** attempted.
- Planned lever (blocked): resume incomplete ext5k dream PPO 2600→5000 under
  frozen r4 WM for seeds 42/314/2718 (`--init-from`, no warm-start).
- Volume audit confirms ext5k seeds still at step **2600**/5000; completed
  true-ballless policies remain at **2500**; WM ensemble @**23000**/coef=12.
- Impact: cannot advance dream training, policy-trust, or development panel.
  Prior development leader (**19%** seed314) retained vs 12% dream-v2 control.
  Not promoted; final panel unused; demo unchanged; prior 17.5% trust-failed
  stochastic policy not exported.
- Action: hard-stop recorded; paid launches not started. Resume next when
  spend limit clears.

## Modal spend limit — daily resume blocked before ext5k finish — 2026-11-08
- Phase: daily honest research cron on
  `cursor/pong-world-model-research-aefe` (FF from latest campaign tip
  `cursor/world-model-research-loop-82da@8692ea5`; base
  `cursor/honest-research-cycle-c57f`).
- Auth gate: credentials present; `modal token set --verify` **OK**; profile
  `ani-sivaa`. Env `MODAL_CREDIT_BALANCE_USD=25` is start telemetry only.
- GPU smoke **FAIL** spend limit; CPU remote hello **FAIL** spend limit.
  Volume reads **OK** (direct `modal.Volume.read_file`).
- Billing: metered ≈ **$30.88** / credits applied ≈ **$30** / billed ≈ **$0.07**.
- Binding pretrust still held from prior true-ballless repair (util PASS
  3.61e-5; ballless PASS 0.0476) — **no collect→WM reburn** attempted.
- Planned lever (blocked): resume incomplete ext5k dream PPO 2600→5000 under
  frozen r4 WM for seeds 42/314/2718 (`--init-from`, no warm-start).
- Volume audit confirms ext5k seeds still at step **2600**/5000; completed
  true-ballless policies remain at **2500**; WM ensemble @**23000**/coef=12.
- Impact: cannot advance dream training, policy-trust, or development panel.
  Prior development leader (**19%** seed314) retained vs 12% dream-v2 control.
  Not promoted; final panel unused; demo unchanged; prior 17.5% trust-failed
  stochastic policy not exported.
- Action: hard-stop recorded; paid launches not started. Resume next when
  spend limit clears.
