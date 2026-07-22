# WorldModel — a Pong world model as a gym for agents

A miniature of the "world model as gym" idea, built end-to-end in one overnight
autonomous run: a from-scratch 64×64 Pong, a learned next-frame world model
that stays coherent over 60+ autoregressive steps, a PPO baseline trained on
real Pong, a dream agent trained **entirely inside the world model's
imagination**, a transfer study (with a cleanly characterized
imagination-exploitation result), and a browser demo where you play against
either agent via client-side ONNX inference.

**Start here: `RESULTS.md`** (every number and artifact) and **`DECISIONS.md`**
(why everything is the way it is). Failures and fallbacks: `FAILURES.md`.
Cross-component contracts: `INTERFACES.md`.

## Play the demo

```bash
cd web && python -m http.server 8321   # then open http://localhost:8321
```

W/S or ↑/↓ to move. Dropdown switches the opponent: `baseline` (strong,
trained on real Pong) vs `dream` (trained only in imagination — losing to it
is hard; that gap is the point).

## Headline results

- **World model drift**: per-pixel MSE .00018 (1 step) → .00386 (60 steps of
  pure dreaming) — linear growth, no collapse. `results/drift_curve_main.png`,
  real-vs-dream GIFs in `results/rollouts/`.
- **Data ablation**: rollout coherence saturates at ~250k transitions (fixed
  6k-step budget); 10% of the data shows a memorization signature.
  `results/ablation/`.
- **Transfer**: baseline +0.12 mean point (52.5% wins) vs dream agent −0.82
  (9%). Inside its dream the agent scores +0.134 — an ~8× flattery factor,
  characterized as distribution-shift exploitation in `results/exploit/`.

## Repo map

```
config.py            every hyperparameter and env constant (single source of truth)
pong/                vectorized Pong env + data collection (pure numpy)
wm/                  world model, trainer (k-step unroll), autoregressive rollout
agents/              shared policy net, PPO (3a), dream training (3b), transfer eval (3c)
evalutils/           plots, GIFs, metrics
infra/               Modal remote runner + local fallback + artifact fetch
web/                 browser demo (vanilla JS + onnxruntime-web)
scripts/             smoke tests, exports, ablation driver, exploit characterization
results/             all artifacts; checkpoints/ holds model weights (gitignored)
```

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/smoke_test.py                  # 2-min end-to-end check
.venv/bin/python -m pong.collect --scale local          # 100k transitions
.venv/bin/python -m wm.train --scale local              # world model
.venv/bin/python -m wm.rollout --scale local --ckpt checkpoints/wm_v1.pt
.venv/bin/python -m agents.train_ppo --scale local      # baseline
.venv/bin/python -m wm.train --scale local --v2 --init-from checkpoints/wm_v1.pt
.venv/bin/python -m agents.train_dream --scale local    # dream agent
.venv/bin/python -m agents.evaluate --scale local \
    --policies baseline=checkpoints/ppo_baseline.pt dream=checkpoints/dream_agent.pt
```

`--scale full` reproduces the overnight numbers; add Modal (`infra/remote.py`)
for GPU: `.venv/bin/python -m infra.remote --module wm.train --gpu -- --scale full`.
