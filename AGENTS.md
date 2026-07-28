# AGENTS.md

## Cursor Cloud specific instructions

This is a Python (3.12) ML/research repo — a from-scratch 64×64 Pong, a learned
world model, PPO + "dream" agents, and a browser demo. See `README.md`,
`RESULTS.md`, and `DECISIONS.md` for the full picture; `INTERFACES.md` for
cross-component contracts.

### Environment
- Dependencies install into a `.venv` at the repo root (`pip install -r requirements.txt`).
  The startup update script keeps `.venv` in sync. Always invoke tools via
  `.venv/bin/python` (e.g. `.venv/bin/python -m wm.train ...`), matching `README.md`.
- `config.py` is the single source of truth for every constant/hyperparameter and
  the `--scale {smoke,local,full}` presets. Default scale is `local`.
- `data/`, `checkpoints/`, `*.pt`, and `results/**/*.npy` are gitignored and are
  generated on demand. No `.pt` checkpoints ship in the repo.

### Two things to run
- **Web demo (interactive product):** `cd web && python3 -m http.server 8321`, then
  open `http://localhost:8321`. It is fully self-contained — `web/model_baseline.onnx`,
  `web/model_dream.onnx`, and `web/constants.js` are committed, so no training/export is
  needed. Non-obvious caveats: it must be served over HTTP (not `file://`), and it loads
  `onnxruntime-web` from `cdn.jsdelivr.net`, so it needs outbound internet at runtime. If
  the CDN or a model file is unreachable it degrades to a scripted opponent with an amber
  banner (not a crash).
- **Pipeline sanity gate:** `.venv/bin/python scripts/smoke_test.py` — ~30–60s CPU
  end-to-end check that prints `SMOKE_LOCAL_OK`. Note it deliberately uses inlined code
  (not the real modules) and it rewrites the committed `results/smoke_policy.onnx`; discard
  that change with `git checkout -- results/smoke_policy.onnx` after running.

### Pipeline (all optional, CPU-runnable at `--scale local`/`smoke`)
`pong.collect` → `wm.train` (+`--v2 --init-from`) → `wm.rollout` →
`agents.train_ppo` → `agents.train_dream` → `agents.evaluate`. `--scale full`
reproduces the overnight numbers and expects Modal GPU via `infra/remote.py`
(needs Modal auth); `infra`/`modal` paths are not required for local dev.

### Lint/tests
There is no configured linter and no automated test suite/framework in this repo;
`scripts/smoke_test.py` is the closest thing to an end-to-end test gate.
