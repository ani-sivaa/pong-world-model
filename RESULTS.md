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

## Phase 2 — world model
- Drift curve (prediction error vs. rollout step): pending → `results/drift_curve.png`
- Real-vs-dreamed GIFs: pending → `results/rollouts/`
- Training-set-size ablation: pending → `results/ablation/`

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

## Phase 3b — dream agent
(pending)

## Phase 3c — transfer (HEADLINE)
- Dream agent: score inside the dream vs. score on real Pong — the gap is the headline number. (pending)

## Phase 4 — web demo
(pending)
