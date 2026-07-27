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
