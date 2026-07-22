# RESULTS.md — every number, chart path, and surprise

## Preflight — PASSED (2026-07-22, ~00:45 PT)
- Local end-to-end (CPU): 500 transitions collected; 30s WM train 1221 steps,
  BCE 0.7179 → 0.0038; 100 policy-gradient steps; ONNX export parity
  max|torch−ort| = 8.9e-08. All 4 stages green in 35s.
- Modal GPU: Tesla T4, torch 2.13.0+cu130, 20 train steps in 2.10s, volume
  write + commit verified. `MODAL_SMOKE_OK`.
- Two infra failures found & fixed before real work (see DECISIONS.md):
  ignore-pattern bug uploading the venv, and repo mount path vs remote sys.path.

## Phase 1 — environment & data
(pending)

## Phase 2 — world model
- Drift curve (prediction error vs. rollout step): pending → `results/drift_curve.png`
- Real-vs-dreamed GIFs: pending → `results/rollouts/`
- Training-set-size ablation: pending → `results/ablation/`

## Phase 3a — baseline agent
(pending)

## Phase 3b — dream agent
(pending)

## Phase 3c — transfer (HEADLINE)
- Dream agent: score inside the dream vs. score on real Pong — the gap is the headline number. (pending)

## Phase 4 — web demo
(pending)
