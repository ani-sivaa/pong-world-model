# RESULTS.md — every number, chart path, and surprise

## Preflight — PASSED (2026-07-22, ~00:45 PT)
- Local end-to-end (CPU): 500 transitions collected; 30s WM train 1221 steps,
  BCE 0.7179 → 0.0038; 100 policy-gradient steps; ONNX export parity
  max|torch−ort| = 8.9e-08. All 4 stages green in 35s.
- Modal GPU: Tesla T4, torch 2.13.0+cu130, 20 train steps in 2.10s, volume
  write + commit verified. `MODAL_SMOKE_OK`.
- Two infra failures found & fixed before real work (see DECISIONS.md):
  ignore-pattern bug uploading the venv, and repo mount path vs remote sys.path.

## Phase 1 — environment & data — DONE
- data/full: T=1,000,000 transitions, 2,068 episodes (2,004 true ends), mean
  episode length 483.6, reward sum 1,402.4 (points +29/−24 — scoring is rare;
  see DECISIONS.md), collected in 22.3s (disk-bound; pure env ~460k tps).
- data/local: T=100,000 (257 eps); data/smoke: T=5,000.
- Determinism: two seed-7 runs bit-identical over 500 steps. Frames exactly {0,255}.
- Sample play GIFs: results/env_samples/.

## Smoke-scale pipeline (pre-GPU gate) — ALL GREEN
- WM v1 (200 steps): val frame loss 0.0287; drift MSE 0.0025@1 → 0.0153@20.
- WM v2 heads trained; PPO 1 iter OK (truncation-bootstrap path exercised);
  dream loop OK (alive-mass 31.2/40 — done head soft-masking behaves);
  transfer eval + exploit-flag logic OK.

## Phase 2 — world model — TRAINED + EVALUATED
- Training: 18,000 steps (6.3k @ k=1 warmup, 11.7k @ k=5 unroll-with-gradient),
  batch 128, T4, 40.5 min. Final val weighted-BCE **0.00184**.
- **Drift curve (the key experiment)** — per-pixel MSE of pure autoregressive
  dreams vs reality, mean over 32 held-out windows, logged actions replayed:
  | rollout step | 1 | 5 | 10 | 15 | 30 | 45 | 60 |
  |---|---|---|---|---|---|---|---|
  | MSE | .00018 | .00027 | .00036 | .00054 | .00134 | .00278 | .00386 |
  Growth is ~linear (ball positional drift), NOT exponential — no model
  collapse over the full 60-step horizon. For scale: two unrelated frames
  differ by ≈ .025. → `results/drift_curve_main.png`, `results/drift_main.json`
- **Real-vs-dream GIFs** (h = 15/30/60, 3 samples each):
  `results/rollouts/main_h*_sample*.gif` — visual check: walls/paddles
  pixel-perfect, ball crisp (change-weighted loss prevented ball blur-out; the
  only |diff| signal is a small ball-position ghost late in long dreams).
- **Training-set-size ablation** (4 parallel T4 jobs, controlled: identical
  6,000-step budget per fraction; drift on the same 32 held-out windows):
  | fraction | mse@1 | mse@30 | mse@60 |
  |---|---|---|---|
  | 10% (100k) | .00016 | .00212 | .00400 |
  | 25% (250k) | .00020 | .00114 | .00329 |
  | 50% (500k) | .00021 | .00149 | .00460 |
  | 100% (1M)  | .00024 | .00165 | .00370 |
  **Finding:** rollout coherence saturates at ~250k transitions under this step
  budget — 10%→25% halves mse@30 (~2 SEM), while 25/50/100% are statistically
  indistinguishable at every horizon. The 10% model shows a memorization
  signature: lowest *train* loss, worst *rollout* drift.
  → `results/ablation/ablation_drift.png`, `ablation_summary.{json,md}`
  (Ops note: parallel Modal launches tripped an app-creation rate limit; fixed
  with 25s staggering — logged in DECISIONS.ablation-runner.md.)

## Phase 3a — baseline agent — DONE
- PPO, 9M env steps total (3M initial + 6M continuation after plateau at
  −0.34; see DECISIONS.md), 35 min wall on local MPS.
- Training tail (sampling): mean point ≈ +0.03, shaped return +0.27, rallies ~200 steps.
- **Formal eval (greedy, 200 episodes): mean point +0.12 — 52.5% wins,
  40.5% losses, 7% truncations; 2.8 paddle hits/ep; mean episode 217 steps.**
  The baseline beats the scripted tracker opponent.
- Checkpoints: ppo_baseline.pt (9M), ppo_baseline_3M.pt (banked mid-run).
- Curves: results/ppo_train.png, results/ppo_train_log.json; play GIF:
  results/agent_baseline_play.gif.

## Phase 3b — dream agent — TRAINED (inside the world model only)
- WM v2 (reward + done heads): 8k fine-tune steps from v1, T4, 18 min; val
  frame loss **0.00081** (better than v1); rollout coherence unchanged
  (mse@60 0.00393 vs 0.00386).
- Dream training: 2,500 A2C updates × 256 parallel dreams × 40-step horizon
  (≈ 25.6M imagined steps), T4, 71 min. **Zero real-Pong steps.**
- Dream return: ~0 → **+0.120** (≈1.2 imagined paddle-hits/dream); entropy
  1.10 → 0.63; dreams survive 37/40 steps.
  → `results/dream_train.png`, `results/dream_train_log.json`

## Phase 3c — transfer (HEADLINE)
| agent | trained on | real mean point | real win rate | hits/ep | ep len |
|---|---|---|---|---|---|
| baseline | 9M real steps | **+0.12** | 52.5% | 2.80 | 217 |
| dream | 25.6M imagined steps | **−0.82** | 9.0% | 0.20 | 50 |

- **The headline gap: +0.134/dream inside the imagination vs −0.82/episode in
  reality.** Per-step hit rate ≈ 8× overestimated by the dream.
- **Exploit characterization (lockstep protocol** — same policy, same action
  stream fed to dream and reality simultaneously, `scripts/characterize_exploit.py`):
  - Dream-vs-real divergence under POLICY actions: MSE@5 = 0.0043 —
    **16× the honest logged-action drift** (0.00027). The WM is accurate on
    its training distribution and wrong off it.
  - Same actions: WM promises +0.045 reward (0.45 hits)/dream; reality returns
    −0.34 and **0 hits in 32/32 envs**.
  - Visual evidence (`results/exploit/exploit_sample*.gif`): from a serve, the
    velocity-less 4-stack makes the deterministic WM hedge TWO ghost balls;
    dreams resolve the ambiguity arbitrarily and permissively — the policy
    learned to intercept dreamed balls that reality never serves.
  - **Diagnosis: distribution-shift exploitation.** The WM was trained on
    tracker-style behavior; the dream policy's action patterns push it
    off-distribution where its physics get permissive. Classic
    imagination-exploitation, cleanly demonstrated at miniature scale.
  - (The automated exploit_flag in transfer.json used a crude threshold and
    did not fire; this manual characterization supersedes it.)
  → `results/exploit/{policy_drift.png, reward_gap.json, exploit_sample*.gif}`,
    `results/transfer.json`, play GIFs `results/agent_{baseline,dream}_play.gif`

## Iteration 2 — diagnosing & fixing the transfer gap

**4-probe diagnostic fanout** (scripts + numbers in `results/diag/`):
1. **Reward hallucination (confirmed, high).** The reward head is nearly
   perfectly calibrated on-distribution (precision .958 / recall .979,
   MSE 9.9e-6) — but under dream-policy visitation the ball VANISHES from
   dreams (39% of dreams; 29.7% ball-less frames at steps 30–39) and **87% of
   imagined reward events fire in frames containing zero ball pixels**.
   Vanished-ball dreams pay 12× more imagined reward than intact ones.
2. **Event coverage, not state coverage (high).** Paddle-y coverage is fine
   (WM renders any paddle position pixel-perfectly). The real hole: **53
   scoring events in 1M transitions** — the WM never saw a ball pass the
   right paddle. Forced-miss lockstep: ball vanishes 20/20, done head never
   fires (max p=0.0016), WM predicts +0.10 where reality gives −1.0.
   **Missing the ball is free inside the dream.**
3. **Policy converged on a corrupted objective (confirmed).** The agent
   tracks NEITHER real nor dreamed balls (12–15% toward-ball moves vs 33%
   chance) — it learned up-camping ball-dodging because that's what the
   corrupted reward channel actually paid. Not underconvergence.
4. **Horizon is NOT the lever (refuted).** Under dream-policy actions the
   dream is off-manifold from step 1; truncating H would destroy signal
   without adding honesty. The lever is action/state distribution.

**Unified diagnosis:** the agent rationally exploited an un-modeled MISS event:
dodge ball → ball vanishes → no done, no −1, free phantom hits.

**Fixes applied (iteration 2, running):**
- `scripts/collect_mixed.py` — new 1M-transition dataset with dense miss/score
  events: tracker 40% / PPO baseline 35% (sampled) / random 15% / v1 dream
  agent 10% (a concentrated source of concede events). Est. ~5–6k scoring
  events vs v1's 53 (~100×).
- WM v3 trained from scratch on mixed data + reward/done heads.
- `--ball-guard` in dream training: reward AND continuation zeroed in
  ball-less dreamed frames — a dream without a ball is over; vanish-states
  pay nothing even if v3 retains rare failure modes.
- Dream agent v2 retrained inside WM v3 (H=40 kept, per probe 4).
**Iteration-2 model-level results (the fix worked):**
- WM v3: 18k steps on mixed data, val frame loss 0.00005; drift on mixed-data
  val windows mse@1 = 0.00001 → mse@60 = 0.00555 (`results/drift_curve_v3.png`).
- **Forced-miss lockstep, v2 vs v3h** (`results/diag/paddlecov_miss_results.json`):
  | metric (guaranteed miss) | WM v2 | WM v3h |
  |---|---|---|
  | predicted cum reward (real −1.0) | +0.10 | **−0.93** |
  | done-head max prob | 0.0016 | **0.9998** |
  Intercept control: 70% correct bounce-backs, honest reward (−0.24 pred vs
  −0.175 real). **Missing now costs −1 and ends the dream — the exploit is dead.**
- Ops: three T4 preemptions/cancellations during this chain (FAILURES.md);
  dream v2 checkpoint recovered at update 1,100/2,500 and continued.

**Iteration-2 agent-level results (FINAL, 2,500 updates in the honest WM):**
| real Pong, 200 eps | dream v1 | dream v2 | baseline |
|---|---|---|---|
| mean point | −0.82 | **−0.76** | +0.12 |
| win rate | 9.0% | **12.0%** | 52.5% |
| paddle hits/ep | 0.195 | **0.62** (3.2×) | 2.80 |
| episode length | 50.2 | **80.4** (+60%) | 217 |
- Dream-internal return collapsed from +0.134 (v1, mostly hallucinated) to
  +0.022 (v2, earned) — the dream no longer flatters much; what the agent
  believes and what reality delivers are far closer.
- **Takeaway:** killing the exploit produced an agent that genuinely plays
  (tracks, intercepts, rallies 60% longer) instead of dodging, but one honest
  Dreamer iteration at this scale does not reach real-data parity. The
  remaining gap is RL sample-efficiency inside honest dreams (every hit must
  be earned; misses cost −1 immediately) plus residual WM imperfection.
  Next levers, if continued: more dream-train compute, additional
  collect→retrain iterations with the improving agent, stochastic latents.
- Demo now ships dream **v2** as the "dream" model (ONNX parity 2.3e-05).

## Iteration 3 — trustworthy dream-agent flywheel (July 2026)

**Protocol.** Policy gradients still came only from imagined rollouts. Real
`VecPong` was used for data acquisition and the fixed 200-episode transfer
evaluation, never for policy updates. The full combined round was launched with:

```bash
python -m scripts.run_flywheel --execute --backend modal --scale full \
  --mode combined --tag flywheel_full_v1 --final-retrain --stochastic
```

The local preflight also ran all independent control paths (`balanced-only`,
`ensemble-only`, `pessimism-only`, and `combined`) at two-step smoke scale.
Those smoke controls validate plumbing only; their deliberately untrained
models are not evidence for causal attribution.

**Data and compute.**
- Initial adaptive set: 1,000,000 real-only transitions; 9,574 hits, 1,199
  scores, 7,589 concedes, and 780 natural truncations. Composition was 30%
  tracker, 20% baseline, 15% current dream, 10% random, 15% engineered rare
  events, and 10% untouched natural holdout.
- Red-team policy: 2,500 imagined-policy updates against three frozen WMs.
  Its one million-transition real replay set was 35% red-team actions and
  increased concedes to 14,436. Every stored frame remained real; ensemble
  disagreement and real-vs-dream mismatch affected priorities only.
- Six deterministic ensemble members (three before and three after red-team
  collection) each completed 18,000 WM steps. Final validation frame losses
  were 0.00535, 0.00569, and 0.00490. The stochastic CVAE ablation completed
  18,000 steps with final validation frame loss 0.00217.
- Both policies completed the configured 2,500 imagined updates through
  checkpointed continuations. The first sequential Modal round took 9.54
  hours; preemptions and client disconnects required resumable follow-up
  segments. No policy continuation used real-environment gradients.

**Trust results.**
- The deterministic ensemble passed every full-scale gate: natural 60-step
  rollout MSE 0.00613; current-policy lockstep MSE 0.00315; dream/real reward
  gap 0.00499; reward MAE 0.00131; done Brier 0.00373; forced-scenario MSE
  0.00374; ball-less positive-reward rate 1.20% (1/83); and uncertainty/error
  rank correlation 0.507 (high-error AUROC 0.765).
- Forced intercept and miss tests produced the correct real events. On the
  forced miss, the deterministic ensemble's terminal prediction included a
  large negative reward (approximately −0.73), rather than the old free miss.
- The stochastic single-model ablation **failed** trust despite lower frame
  drift: done Brier 0.251, concede reward MAE 1.003, score reward MAE 0.984,
  and no usable epistemic uncertainty ranking. Its sampled-latent variation
  was also nearly collapsed. It is rejected in favor of the deterministic
  ensemble.

**Final 200-episode real transfer (same seed/protocol as v2):**

| real Pong | dream v2 control | deterministic ensemble | stochastic ablation |
|---|---:|---:|---:|
| win rate | 12.0% | **8.5%** | **17.5%** |
| mean point | −0.76 | −0.83 | −0.65 |
| paddle hits/episode | 0.62 | **1.085** | 1.00 |
| episode length | 80.4 | **115.7** | 104.8 |
| trust gate | not measured by new suite | **PASS** | **FAIL** |

**Conclusion: honest negative result.** The trustworthy deterministic agent
rallies longer and hits more often than v2, but its 8.5% win rate is below the
12% control and the >20% target. The stochastic policy reaches 17.5%, but its
reward/done model is badly miscalibrated, so that apparent gain is not accepted
and no web export was performed. The flywheel fixed simulator trust and made
uncertainty predictive; it did not yet improve honest match wins. A next round
should feed the stochastic model's score/concede failures back into acquisition
or improve its conditional reward/done heads before more policy training.

Artifacts: `results/{transfer,trust}_flywheel_full_v1_combined*.json`,
`results/flywheel_full_v1_{initial,enriched}_meta.json`, and
`results/flywheel_full_v1_flywheel_state.json`.

## Honest-agent round two — implementation and smoke evidence (2026-07-27)

No full-training or transfer result is claimed in this section.

- Added real-only terminal context anchors (`hit`, `score`, `concede`, `miss`,
  and truncation), backwards-decayed priorities, an overlap-safe event bitmask,
  and an untouched natural holdout. A 256-transition local smoke wrote only
  binary real frames and passed collector boundary/self checks across 24
  streams.
- Stochastic latents now affect frame decoding only. Reward/done heads use the
  deterministic pre-latent state/action context. Existing deterministic,
  headless, and stochastic checkpoint keys remain loadable. Training adds
  event-balanced outcome losses, KL warmup, prior/posterior KL, prior
  variation, and latent-utilization logs.
- Trust now fail-closes stochastic promotion on missing or collapsed prior
  standard deviation, sampled-frame variation, latent utilization,
  posterior/prior KL, and two-direction coverage from the same ambiguous serve
  context. Deterministic thresholds are unchanged.
- `agents.train_dream --algorithm ppo` adds clipped imagined PPO with soft-done
  GAE. The WM is frozen, all imagined targets are detached, pessimism and the
  ball guard remain active, and runtime/tests reject any WM gradient.
- The local collect → one-step stochastic WM → one-update/two-step PPO → trust
  smoke completed. Its smoke-scale trust report passed the intentionally loose
  preflight thresholds; this is plumbing evidence, not model-quality evidence.
- 24 unit/integration tests passed, including real-only context boundaries,
  latent-invariant heads, balanced loss weights, collapse rejection, GAE,
  clipped PPO with zero WM gradients, immutable final-panel selection, and
  dry-run orchestration. `compileall` and `git diff --check` also passed.
- The campaign preregisters acquisition seeds `(1103,2207,3301,4409,5519)`,
  WM/policy seeds, development seeds `(7001,7003,7013)`, and one untouched
  final panel `(104729,130363,155921)` at 200 episodes per seed. Development
  evidence alone chooses a candidate. The final panel is immutable,
  overwrite-protected, reports Wilson 95% intervals, and can be consumed once.
- The resumable controller records commands, dataset composition/event counts,
  seeds, trust/development evidence, retries, wall time, and unavailable cost
  fields in a structured manifest. It uses a process lock and bounded
  rounds/retries, has no export stage, and starts a new acquisition round
  before more policy optimization whenever a trust gate fails. Round zero also
  preregisters a same-seed REINFORCE development ablation for PPO attribution.

Earlier blockers (kept for history): Modal billing previously returned
`Token missing` when credentials were not injected into the running Cloud
Agent. No paid jobs launched under that blocked state; final panel unused.

### Honest research cycle resume (2026-08-03) — auth cleared
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` present; Modal profile `default` ok; billing
  summary JSON ok (spend fields only — remaining credit from env = **$30.0**).
- Manifest status cleared from `blocked_current_run_environment_not_refreshed`
  → `running`; new `resume_attempt` appended.
- Volume recovery: five prior full-scale rounds (`r0`–`r4`) already on
  `worldmodel-vol` with collect+WM+pretrust only (no policies). All five
  failed pretrust; binding gate `stochastic_latent_utilization` (~4e-8–8e-8
  vs 1e-7); often also `ballless_positive_rate`. Best near-miss: **r3**
  (ballless passed; latent util 7.79e-8). Evidence under
  `results/campaign_evidence/round-*_{pretrust,collect_meta}.json`.
- This cycle: full scale (`1e6` transitions / `18k` steps / `2500` updates),
  `--max-rounds 2`, stop early if trust+transfer teach nothing new. Demo
  export deferred until trust-passing promotion.
- Control baseline unchanged: dream v2 ~12% wins; trust-passing deterministic
  ensemble retained; prior 17.5% trust-failed stochastic policy not exported.

```bash
python3 -m scripts.run_honest_campaign --execute \
  --tag honest_campaign_v1 \
  --base-wm \
    /vol/checkpoints/flywheel_full_v1_combined_final_wm0.pt \
    /vol/checkpoints/flywheel_full_v1_combined_final_wm1.pt \
    /vol/checkpoints/flywheel_full_v1_combined_final_wm2.pt \
  --redteam /vol/checkpoints/flywheel_full_v1_combined_redteam.pt \
  --stochastic-policy \
    /vol/checkpoints/flywheel_full_v1_combined_stochastic_dream.pt \
  --transitions 1000000 --steps 18000 --updates 2500 \
  --max-rounds 2 --retries 2
```

Limitations: the 80% development threshold is deliberately demanding and is
not required for cycle success; Wilson intervals quantify evaluation
uncertainty but do not remove dependence among episodes from the same
vectorized simulator; Modal's billing summary may expose spend without an
explicit remaining-credit field (env balance used here); and a below-target
result on the one final panel terminates the protocol because tuning after
reading that panel would invalidate its untouched status.

### Honest research cycle complete (2026-08-10) — ≤2 rounds, no promotion
Primary success for this cycle is transferable evidence + lab notebook, not
an 80% win-rate requirement.

- **Credits:** Modal billing summary at notebook time ≈ metered $4.43 /
  credits applied ≈ $3.42 / billed $0.00. Env `MODAL_CREDIT_BALANCE_USD=30`
  is static start telemetry, not live remaining balance.
- **Rounds completed:** 2 / 2 (`max_rounds=2`). Both stopped at **pretrust**
  before any dream-policy training.
- **Trust:**
  - r0 (default mix): **FAIL** —
    `ballless_positive_rate` 0.066 > 0.05;
    `stochastic_latent_utilization` 8.25e-8 < 1e-7.
  - r1 (collapse mix, stochastic=.45): **FAIL** —
    `ballless_positive_rate` 0.273 > 0.05;
    `stochastic_latent_utilization` 4.38e-8 < 1e-7.
  - Other gates passed in both rounds (reward/done calibration, rollouts,
    prior std, serve-direction coverage, etc.).
- **Development win rates vs 12% dream-v2 control:** n/a — no policies
  trained; no development panel; no transfer delta this cycle.
- **What the WM still gets wrong:** latent codes remain near-collapsed under
  the utilization gate; ballless frames still predict ball mass too often.
  Collapse-mix acquisition made ballless worse and did not lift latent
  utilization over threshold (same binding failures as prior volume history).
- **final_evaluation_uses:** 0 (never promoted).
- **Promotion:** **not promoted.** Demo/web export **unchanged** (trust
  gates never passed).
- **Controls retained:** dream v2 ~12%; trust-passing deterministic ensemble;
  prior 17.5% trust-failed stochastic policy not exported.
- Evidence: `results/campaign_evidence/this_cycle_r{0,1}_{pretrust,collect_meta}.json`,
  WM train logs `wm_log_r*_wm*.json`, manifest
  `results/honest_campaign_v1_campaign_manifest.json`.

### Honest research cycle (2026-08-13) — binding repairs + trust-pass flywheel
Recovered onto this lineage from prior campaign branch work:

- Binding-trust repair package: `latent_inject_gain=2.5`, relative utilization
  hinge (`utilization_coef=0.15`, target `5e-6`), `ballless_reward_coef=3.0`,
  `acquisition_mix_binding_repair`, CAPS `wm_train=10800` / `dream=14400`.
- `honest_campaign_v2` outcomes:
  - r0 pretrust: ballless FAIL (util cleared).
  - r1 pretrust **PASS** → dream PPO → development best **9.33%** wins
    (below **12%** dream-v2 control; not eligible / not promoted).
  - r2 pretrust **PASS** (util **4.74e-5**, ballless **0.033**).
  - r2 `policy-0` reached ~180/2500 imagined updates then Modal
    `ResourceExhaustedError` (workspace spend limit). Partial checkpoint
    on volume at **step 100**.
- **Not promoted.** Demo/web export unchanged. Final panel unused.

### Honest research cycle (2026-10-12) — resume after spend-limit clear
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; Modal client_ok; profile `default`.
- Env credit telemetry = **$25**. GPU smoke **PASS**: Tesla T4,
  torch 2.13.0+cu130, 20 train steps in 1.747s
  (`results/campaign_logs/modal_smoke_2026-10-12.log`).
- Binding repairs + resume hardening ported to this branch; unit tests 24/24.
- **Did not** re-burn collect→WM on the old v1 binding failures. Resume target:
  `honest_campaign_v2` round-2 `policy-0` with `--init-from`
  `/vol/checkpoints/honest_campaign_v2_r2_ppo_seed42.pt` (step 100).
- Trust gates (latest): r2 pretrust PASS (util 4.74e-5, ballless 0.033).
- Transfer vs 12% dream-v2 control: r1 best **9.33%** (below control);
  round-2 policy incomplete → no new development panel yet.
- Promotion: **not promoted** pending round-2 policy completion. Demo/web
  export unchanged. Final panel unused. Prior 17.5% trust-failed stochastic
  policy not exported.
- **Round-2 policy-0 complete (2026-10-12):** wall-clock stop at update
  **2302/2500** (`DREAM_OK`, ckpt on volume). Mid-job Modal restart had
  reloaded the step-1000 periodic checkpoint with a fresh 14400s window.
- **policy-trust-0:** **PASS** (overall_pass=True).
- **development-0** (seeds 7001/7003/7013, 100 eps): mean win rate
  **0.0%** (Wilson pooled CI [8.673617379884035e-19, 0.012642971224546034]);
  mean hits ≈ 0.33/ep; mean point −1.0. **Below** the 12% dream-v2 control
  and below r1 best 9.33%.
- **Live:** round-2 `policy-1` (seed 314) in flight (~update 400/2500);
  no demo export; final panel unused; prior 17.5% trust-failed stochastic
  policy not exported.

### Honest research cycle (2026-10-13) — continue r2 policy-1
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; Modal profile `default` ok.
- Env credit telemetry = **$25** (start-of-run static). Billing snapshot:
  metered ≈ $11.51 / credits applied ≈ $9.89 / billed $0.00 (not a live
  remaining-credit field).
- GPU smoke **PASS**: Tesla T4, torch 2.13.0+cu130, 20 train steps in 1.285s
  (`results/campaign_logs/modal_smoke_2026-10-13.log`).
- Binding repairs already trusted on this lineage (r1/r2 pretrust PASS). **Did
  not** re-burn collect→WM for the old v1 latent-collapse / ballless pair.
- Volume resume: `honest_campaign_v2_r2_ppo_seed314.pt` at **step 1200/2500**.
  Controller now `--init-from` that checkpoint when a partial volume artifact
  exists without a recorded stage failure (mid-flight controller exit).
- **Round-2 policy-1 complete (2026-10-13):** resumed from update 1200 →
  **2500/2500** (`DREAM_OK`, wall ≈ 12412s, final dream_return −0.051).
- **policy-trust-1:** **PASS** (overall_pass=True; 0 failed gates).
- **development-1** (seeds 7001/7003/7013, 100 eps): mean win rate **0.0%**
  (Wilson pooled CI same floor as development-0); mean point −1.0; hits
  ≈ 0.11/ep. **Below** the 12% dream-v2 control and below r1 best 9.33%.
- **Round-2 policy-2 complete (wall-stop):** seed 2718 from scratch → update
  **1465/2500** (`DREAM_OK` after `WALL-CLOCK CAP`, wall ≈ 14411s,
  dream_return −0.057). Volume checkpoint retained for a later `--init-from`.
- **policy-trust-2:** **PASS**. **development-2:** mean win rate **0.0%**
  (hits ≈ 0.5/ep; mean point −1.0). Below the 12% dream-v2 control.
- **Selection:** `eligible=false` (all three r2 candidates at 0% development
  wins; each strict-trust PASS). Controller status
  `development_threshold_not_met_before_round_limit`.
- **Credits (end of cycle):** Modal billing metered ≈ **$19.10** / credits
  applied ≈ **$17.35** / billed **$0**. Env `MODAL_CREDIT_BALANCE_USD=25`
  remains start telemetry only.
- **Transfer vs 12% dream-v2 control:** r1 best **9.33%** still leads; r2
  development panel is **0% / 0% / 0%** for seeds 42 / 314 / 2718.
- **Promotion:** **not promoted.** Demo/web export **unchanged.** Final
  panel unused (`final_evaluation_uses=0`). Prior 17.5% trust-failed
  stochastic policy not exported.
- Trust (latest): r2 pretrust PASS (util 4.74e-5, ballless 0.033); all three
  r2 policy-trust reports PASS. Binding repairs held; no collect→WM reburn.

### Honest research cycle (2026-10-14) — continue after r2 development collapse
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; Modal profile `default` ok; client_ok.
- Env credit telemetry = **$25** (start-of-run static). Billing snapshot at
  start: metered ≈ $18.97 / credits applied ≈ $17.22 / billed $0.00.
- GPU smoke **PASS**: Tesla T4, torch 2.13.0+cu130, 20 train steps in 1.655s
  (`results/campaign_logs/modal_smoke_2026-10-14.log`).
- Continued from latest campaign lineage (`honest_campaign_v2` on prior daily
  branch tip): binding repairs already clear pretrust (r1/r2 util+ballless
  PASS). **Did not** re-burn collect→WM for the old v1 latent-collapse pair.
- Prior cycle end state: r2 selection `eligible=false`; development **0%/0%/0%**
  vs r1 best **9.33%** and the **12%** dream-v2 control; not promoted; final
  panel unused; no demo export.
- Today's concrete lever (transfer, not binding): after trust-pass rounds whose
  development max stays below the 12% control, prefer a transfer-repair
  acquisition mix and warm-start dream PPO weights from the best prior
  development checkpoint (`r1_ppo_seed314`, 9.33%) with a reset update
  counter — then continue the flywheel into round 3.
- Transfer-repair code landed + unit tests green (27/27). Dry-run plan for
  round 3 uses mix `tracker=.10,rare=.35,redteam=.20,stochastic=.25,natural=.10`
  and `--init-from r1_ppo_seed314 --warm-start` on all three dream PPO seeds.
- **Live execute started (2026-10-14):** controller skipped r0–r2 completed
  stages, fetched r1/r2 selections, and launched `round-3:collect` on Modal
  with the transfer-repair mix (confirmed in the live `infra.remote` argv).
  Stochastic collector checkpoint is the r1 9.33% policy. No demo export;
  final panel unused.
- **Round-3 collect complete (~570s wall):** `COLLECT_ADAPTIVE_OK` T=1,000,000,
  real_only=True, transfer-repair mix (`rare=.35`). Evidence:
  `results/campaign_evidence/v2_r3_collect_meta.json`.
- **Live:** `round-3:wm-0` training started (18k stochastic steps, balanced
  sampler + bootstrap). Dream PPO still gated on pretrust.
- **Round-3 wm-0 complete:** `WM_TRAIN_OK` 18,000 steps, final val frame
  **0.00553**, wall ≈ 6971s. Training-time latent util ≈ 2e-4 with ballless
  penalty ≈ 0 (hinge healthy vs the 1e-7 gate). `wm-1` started.
- **Round-3 wm-1 complete:** `WM_TRAIN_OK` 18,000 steps, final val frame
  **0.00769**, wall ≈ 6933s. `wm-2` training started afterward.
- **Round-3 wm-2 complete:** `WM_TRAIN_OK` 18,000 steps, final val **0.00734**,
  wall ≈ 6960s.
- **Round-3 pretrust: FAIL** (no dream PPO; science rule honored).
  - `stochastic_latent_utilization` **PASS** 4.11e-5 ≥ 1e-7
  - `ballless_positive_rate` **FAIL** 0.122 > 0.05
  - Evidence: `results/campaign_evidence/v2_r3_pretrust.json`
- Concrete ballless repair before another collect→WM burn: raise
  `ballless_reward_coef` 3.0→**6.0**, tighten `ballless_mass_tau` 0.5→**0.35**,
  and let the controller select `acquisition_mix_calibration` (rare=.40) for
  round 4 from the ballless failure.
- **Live round-4 execute started:** calibration mix (`rare=.40`) + strengthened
  ballless penalty. Dream PPO remains gated on pretrust.
- **Round-4 collect complete (~639s):** calibration mix confirmed; WM-0 training
  started with `ballless_reward_coef=6.0`.
- **Round-4 wm-0 complete:** final val **0.00696**, util hinge ≈ 1.1e-3,
  ballless train penalty ≈ 0. `wm-1` in flight.

### Honest research cycle (2026-10-15) — finish r4 WM + pretrust
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; Modal profile `default` ok; client_ok.
- Env credit telemetry = **$25** (start-of-run static). Billing snapshot at
  start: metered ≈ **$30.07** / credits applied ≈ **$28.18** / billed **$0**.
- GPU smoke **PASS**: Tesla T4, torch 2.13.0+cu130, 20 train steps in 2.721s
  (`results/campaign_logs/modal_smoke_2026-10-15.log`).
- Continued from latest campaign tip (`honest_campaign_v2` on prior daily
  branch): binding repairs still clear latent util on this lineage; r3 failed
  **only** `ballless_positive_rate` (0.122). Round 4 already collected under
  calibration mix (`rare=.40`) with `ballless_reward_coef=6.0` /
  `ballless_mass_tau=0.35`.
- Volume recovery (2026-10-15):
  - `r4_wm0.pt` **complete** (step 18000); `r4_wm1.pt` **complete** (18000);
  - `r4_wm2.pt` **partial** (step **11000**/18000) — no train log yet;
  - no `trust_honest_campaign_v2_r4_prepolicy.json` yet.
- Concrete repair this cycle (not another blind collect→WM): add WM
  step-resume via `--init-from` (continue saved step counter; no-op when
  already at budget) so the controller finishes `wm-2` without reburning
  `wm-0`/`wm-1` or recollecting. Collect stage recovered into the manifest.
- Dream PPO remains gated on r4 pretrust. Warm-start from r1 seed314 (9.33%)
  stays planned if pretrust passes. No demo export; final panel unused;
  prior 17.5% trust-failed stochastic policy not exported.
- Transfer control unchanged: dream v2 **12%**; r1 best development **9.33%**;
  r2 development **0%/0%/0%**.
- **Live execute (2026-10-15):** controller skipped r0–r3 completed/stopped
  stages; `round-4:collect` recovered SKIP; `wm-0`/`wm-1` no-op resumed in
  ~13s/`~11s` (`already complete at step 18000`); `wm-2` resumed from step
  **11000**/18000 with `ballless_reward_coef=6.0` (train ballless≈0 so far).
  First resume attempt hit a CosineAnnealingLR `initial_lr` KeyError (logged in
  FAILURES.md) and was fixed before relaunch.

- **Round-4 wm-2 complete (resume):** step 11000→**18000**, final val
  **0.00691**, wall ≈ 3966s (`WM_TRAIN_OK`).
- **Round-4 pretrust: FAIL** (no dream PPO; science rule honored).
  - `stochastic_latent_utilization` **PASS** 5.84e-5 ≥ 1e-7
  - `ballless_positive_rate` **FAIL** 0.071 > 0.05 (4 violations / 56
    ballless frames) — improved vs r3's **0.122** but still above gate
  - Evidence: `results/campaign_evidence/v2_r4_pretrust.json`
- Next concrete repair (no new collect→WM burn): raise
  `ballless_reward_coef` 6→**12**, tighten `ballless_mass_tau` 0.35→**0.20**,
  hinge positive reward at the trust threshold (`relu(r-0.05)`), then
  fine-tune the three r4 WMs +3000 steps from the finished checkpoints and
  re-run pretrust.
- **Live ballless fine-tune started (2026-10-15):** same r4 dataset, no new
  collect. Sequential `--init-from` continues each WM 18000→**21000** under
  coef=12 / tau=0.20 / hinge=0.05, then re-pretrust
  (`trust_honest_campaign_v2_r4_ballless_ft_prepolicy.json`). Dream PPO still
  gated. Billing at launch: metered ≈ $31.20 / billed $0.

- **Ballless fine-tune interrupted by Modal spend limit (HARD STOP):**
  - `wm-0`: **21000**/21000 complete under coef=12 / tau=0.20 / hinge=0.05
    (final val **0.00595**).
  - `wm-1`: wall/disabled at step **20000**/21000 (coef=12 applied in ckpt).
  - `wm-2`: still **18000** under prior coef=6 / tau=0.35 (not fine-tuned).
  - Re-pretrust **not run** on the mismatched ensemble.
- End billing snapshot: metered ≈ **$31.94** / billed ≈ **$0.05**. Env
  `MODAL_CREDIT_BALANCE_USD=25` remains start telemetry only.
- **Trust gates (latest completed):** r4 pretrust util **PASS** 5.84e-5;
  ballless **FAIL** 0.071 (near-miss vs r3 0.122). r1/r2 still the last
  trust-passing policy rounds.
- **Transfer vs 12% dream-v2 control:** r1 best development **9.33%** still
  leads; r2 **0%/0%/0%**; r3/r4 produced **no** new policies this cycle.
- **Promotion:** **not promoted.** Demo/web export **unchanged.** Final panel
  unused (`final_evaluation_uses=0`). Prior 17.5% trust-failed stochastic
  policy not exported.
- Manifest status: `budget_or_auth_exhausted`.

### Honest research cycle (2026-10-16) — auth OK, spend limit hard stop
- Continued latest campaign tip onto daily branch
  `cursor/honest-research-cycle-c8aa` from `cursor/honest-research-cycle-3c24`
  (`f318a07`).
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` **present**; `modal token set --verify` **OK**;
  profile `default`; billing summary JSON ok. Env credit telemetry = **$25**
  (start-of-run static only).
- Billing snapshot: metered ≈ **$32.07** / credits applied ≈ **$30** /
  billed ≈ **$0.02**.
- GPU smoke **FAIL** (spend limit): `ResourceExhaustedError: Workspace ... has
  exceeded its spend limit`
  (`results/campaign_logs/modal_smoke_2026-10-16.log`). CPU remote hello also
  blocked by the same spend-limit gate.
- Volume reads still work. Checkpoint audit confirms mismatched r4 ballless
  fine-tune ensemble (no re-pretrust):
  - `r4_wm0.pt` step **21000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm1.pt` step **20000**, coef=12 / tau=0.20 / hinge=0.05
  - `r4_wm2.pt` step **18000**, coef=6 / tau=0.35 (not fine-tuned)
  → `results/campaign_logs/volume_ckpt_audit_2026-10-16.log`
- **No paid jobs launched** this cycle. Did **not** start another collect→WM
  round (binding repair still unfinished; spend limit is the blocker).
- Local repair-code regression: **27** unit tests passed
  (`results/campaign_logs/unit_tests_2026-10-16.log`).
- **Trust gates (unchanged since 2026-10-15):** latest completed r4 pretrust
  util **PASS** 5.84e-5; ballless **FAIL** 0.071. r1/r2 remain the last
  trust-passing policy rounds. Ballless FT re-pretrust still blocked.
- **Transfer vs 12% dream-v2 control:** r1 best development **9.33%** still
  leads; r2 **0%/0%/0%**; r3/r4 still produced **no** new policies.
- **Promotion:** **not promoted.** Demo/web export **unchanged.** Final panel
  unused (`final_evaluation_uses=0`). Prior 17.5% trust-failed stochastic
  policy not exported.
- Manifest status remains `budget_or_auth_exhausted` with a new
  `resume_attempts` entry and hard-stop note for 2026-10-16.
- **Resume next (when spend limit clears):** finish `wm1` 20000→21000 and
  `wm2` 18000→21000 under coef=12 / tau=0.20 / hinge=0.05 on the same r4
  data; re-pretrust; only then dream PPO with warm-start r1 seed314.







## Phase 4 — web demo — COMPLETE (both models)
- Headless-browser verification (Playwright): page loads, ONNX models load
  (no fallback banner), zero console/page errors, simulation advances, agent
  plays — screenshot `results/webdemo_check.png`; verifier:
  `scripts/verify_webdemo.py`.
- **Both agents shipped with a dropdown toggle**: `baseline` (strong) and
  `dream` (weak in reality — playing it IS the transfer-gap exhibit).
  ONNX parity: baseline 1.3e-05, dream 2.7e-05; argmax agreement 8/8 each.
- Client-side inference latency: **~1.2 ms/tick**.
- **Play it: `cd web && python -m http.server 8321` → http://localhost:8321**
  (a server is already running on :8321 from the overnight run).
