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

### Honest research cycle (2026-08-11) — binding-trust repair then flywheel
- Auth gate: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` /
  `MODAL_CREDIT_BALANCE_USD` present; Modal profile ok; billing summary ok
  (metered ≈ $4.53 / credits applied ≈ $3.42 / billed $0.00). Env credit
  telemetry at start of this cycle = **$25**.
- Prior cycle (`honest_campaign_v1`) stopped after ≤2 pretrust failures on
  `stochastic_latent_utilization` + `ballless_positive_rate`. Collapse-mix
  acquisition alone did not repair those gates.
- **Concrete repairs before more collect→WM spend:**
  - Stronger latent residual injection (`latent_inject_gain=2.5`)
  - Weaker KL / longer warmup (`kl_coef=5e-4`, `kl_warmup_frac=0.5`,
    `free_bits=0.08`)
  - Relative utilization hinge (O(1) when collapsed; absolute hinge was
    drowned by frame BCE)
  - Ballless-reward consistency loss (penalize `relu(reward)` when predicted
    interior ball mass ≈ 0)
  - New `acquisition_mix_binding_repair` (rare=.35, stochastic=.25) replaces
    the failure-driven collapse mix when latent utilization fails
- **Local smoke evidence** (`repair_smoke_v2b`, 200 steps, CPU): trust report
  `results/campaign_evidence/repair_smoke_v2b_pretrust.json` — latent
  utilization **1.17e-4** (full gate ≥1e-7), ballless **0.0** (full gate
  ≤0.05), serve-direction coverage 1.0. Prior absolute-hinge smoke util was
  2.8e-8 (would fail full).
- Next: launch `honest_campaign_v2` on Modal (full scale, max_rounds=3) with
  the repaired trainer. Dream-policy training only if pretrust passes. Demo
  export remains gated. Control retained: dream-v2 ~12% wins; prior 17.5%
  trust-failed stochastic policy not exported.
- **First paid Modal job this cycle:** `round-0:collect` for `honest_campaign_v2` launched (default mix). Controller log: `results/campaign_logs/controller_v2.log`.

- **Round-0 WM-0 mid-train (Modal, ~10.8k/18k steps):** relative utilization hinge
  still firing intermittently (`util` loss ~0–0.002) rather than sitting at the
  late-training collapse seen historically; ballless penalty remains tiny
  (~1e-5). Continuing through wm-0/1/2 → pretrust before any dream-policy work.



- **Round-0 WM-0 finished under wall-clock:** `WM_TRAIN_OK` at step **17592/18000**
  (hit previous 7200s cap; val_frame 0.00709). Late-training `util` hinge still
  non-zero at times through 17k — unlike historical collapse-to-~1e-8. Raised
  `CAPS.wm_train` to 10800s for subsequent members. WM-1 training underway.

- **Round-0 WM-1 complete:** full **18000/18000** steps. Train-log latent
  utilization remains ~1e-3 class into the final third (orders of magnitude
  above the 1e-7 gate). WM-2 training underway with raised 10800s cap.

- **Round-0 pretrust (full):** **FAIL** on `ballless_positive_rate` only.
  - `stochastic_latent_utilization` **PASS** at **5.58e-4** (threshold ≥1e-7) —
    binding latent-collapse gate cleared by the repair package.
  - `ballless_positive_rate` **FAIL** at **0.123** (threshold ≤0.05).
  - Other stochastic gates (prior_std, KL, serve coverage, sample MSE) all PASS.
  - No dream-policy training; no demo export.
- **Round-1:** calibration acquisition mix (rare=.40) collected; WM retrain with
  raised `ballless_reward_coef=3.0` (was 0.75). Goal: keep utilization healthy
  while driving ballless under 0.05.

- **Round-1 mid-flight:** wm-0 completed 18k; wm-1 training with strengthened
  ballless penalty (`coef=3.0` + hard interior mask). Credits applied ≈ $12.3 /
  env start telemetry $25. Still no policy training / no demo export.

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
