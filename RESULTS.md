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
- Training-set-size ablation: running (4 parallel T4 jobs) → `results/ablation/`

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
