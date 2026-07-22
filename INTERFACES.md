# INTERFACES.md — cross-agent contracts

Authored by the main thread BEFORE the wave-1 fanout (deviation from the original
plan of eval-utils writing it: parallel agents can't wait on each other, so the
contracts are fixed up front; agents append clarifications under their own
heading, never edit others' sections).

**Rule: no agent edits files outside its ownership row. Cross-boundary needs are
noted here and applied by the main thread.**

## File ownership

| Owner        | Files |
|--------------|-------|
| main thread  | config.py, INTERFACES.md, DECISIONS.md, RESULTS.md, FAILURES.md, wm/*, agents/*, scripts/smoke_*.py |
| env-builder  | pong/env.py, pong/collect.py, scripts/export_constants.py, web/constants.js (generated), data/*, results/env_samples/* |
| eval-utils   | evalutils/*.py |
| web-demo     | web/* except constants.js and *.onnx |
| modal-infra  | infra/* (seeded with infra/modal_common.py by main thread — extend, do not rewrite the image/volume spec: the image cache keys on it) |
| onnx-export  | web/*.onnx, scripts/export_onnx.py |
| ablation-runner | scripts/run_ablation.py, results/ablation/* |

## Env API — `pong/env.py`

Pure numpy (NO torch import). All constants from `config.ENV` — zero literals in code.

```python
class VecPong:
    def __init__(self, n_envs: int, seed: int): ...
    def reset(self) -> np.ndarray                 # uint8 [n, 64, 64], all envs
    def step(self, actions: np.ndarray)           # int array [n] in {0,1,2}
        -> (frames uint8 [n,64,64], rewards float32 [n], dones bool [n],
            info: dict with "paddle_hit" bool [n], "point" int8 [n] in {-1,0,+1},
                  "terminal_frame" uint8 [n,64,64])
    # done envs AUTO-RESET inside step(); the returned frame for a done env is
    # the FIRST frame of the new episode; reward/done describe the ended one.
    # info["terminal_frame"][i] = the TRUE final frame of the ended episode
    # (post-step, PRE-reset) for done envs; zeros for non-done envs. Needed so
    # PPO can bootstrap the value of truncated (max_steps) episodes correctly
    # — "point"==0 with done=True identifies truncation vs a real terminal.

class Pong:            # thin n=1 wrapper, same semantics, unbatched returns
def scripted_action(state..., eps_random, rng) -> actions  # the collection policy
```

Physics contract (must match web/pong.js EXACTLY — see web section):
- float positions; render with `int(floor(x + 0.5))` (NOT Python round()).
- walls: rows 0 and 63 white; ball top-left y reflected at 1 and 61 (`y=2-y` / `y=122-y`), vy flips.
- paddles: left cols [2,4), right cols [60,62), height 12, top-left y clamped to [1, 51].
- ball 2×2, |vx|=1.5 constant; bounce iff crossing test passes:
  `prev_edge <= plane < new_edge` AND vertical overlap (`by+2 > py and by < py+12`).
  Right plane: ball right edge vs x=60; left plane: ball left edge vs x=4 (reflect).
  On bounce: reflect x about plane, vx=-vx, vy = clip((ball_cy-paddle_cy)/(paddle_h/2),-1,1)*max_vy.
- score when ball fully off-screen: x > 63 (left scores, agent reward -1) or x+2 < 0 (agent +1).
- serve: ball centered; vx sign = seeded RNG; vy = seeded choice from ENV["serve_vy_choices"].
- actions: 0=stay, 1=up(y-=speed), 2=down(y+=speed). Agent = RIGHT paddle.
- rewards: r_score / r_concede / r_hit from config; episode done on point or max_steps.

STEP ORDER (identical in Python and JS — parity depends on it):
1. paddles move: agent by action, opponent scripted tracker (move opp_speed
   toward ball CENTER y iff |paddle_cy - ball_cy| > opp_deadzone); clamp y to [1, 51].
2. ball moves: x += vx; y += vy   (store prev x before the move for crossing tests)
3. wall bounce: reflect y at 1 / 61, flip vy.
4. paddle collision (crossing test vs PREV x, paddles at their NEW y):
   reflect x about plane, vx = -vx, vy = clip(offset/(paddle_h/2), -1, 1)*max_vy;
   agent-paddle contact => reward += r_hit, info paddle_hit.
5. scoring: ball fully off-screen => point, done.
6. t += 1; t >= max_steps => done (point 0).
7. render.
Serve/reset state: ball top-left (31, 31); paddles y = 26 (centered); vx sign
and vy from seeded RNG as specified above.
Scripted collection policy (agent paddle): with prob eps_random uniform random
action, else tracker with opp_deadzone: paddle_cy < ball_cy - dz -> 2 (down),
> ball_cy + dz -> 1 (up), else 0.

## Data format — written by `pong/collect.py` to `data/<scale>/`

Contiguous streams, episodes back-to-back, frames stored ONCE and indexed:
- `frames.npy`  uint8 [T, 64, 64] — frame BEFORE each action (frames[t] ↔ actions[t]); the frame after action t is frames[t+1] **unless dones[t]** (then frames[t+1] is the next episode's first frame).
- `actions.npy` uint8 [T]; `rewards.npy` float32 [T]; `dones.npy` bool [T]
- `meta.json`: {seed, T, n_episodes, env_constants: config.ENV, collect: config.COLLECT}

Training-side sampling (main thread owns): a (stack, action, next) sample at index t
uses frames[t-3..t] → frames[t+1]; valid iff no done in [t-3, t] and t-3 ≥ episode start.
k-step windows need no done in [t-3, t+k-1].

## evalutils API — importable as `from evalutils import plots, gifs, metrics`

Deps: numpy, matplotlib (Agg backend, headless), imageio, pillow only.

```python
plots.line_plot(xs, ys: dict[str, sequence], title, xlabel, ylabel, out_path,
                log_y=False)                      # one PNG, legend from keys
gifs.save_gif(frames: uint8 [T,H,W] or [T,H,W,3], out_path, fps=30, scale=4)
                                                  # nearest-neighbor upscale
gifs.side_by_side(clips: list[uint8 [T,H,W]], labels: list[str], out_path,
                  fps=30, scale=4)                # horizontal montage + text labels,
                                                  # clips may differ in T (pad w/ last frame)
metrics.mse_curve(real: uint8 [T,H,W], pred: uint8 [T,H,W]) -> float32 [T]
                                                  # per-step MSE on [0,1]-normalized pixels
metrics.summarize_episodes(points: int8 [N], hits, lengths) -> dict  # means etc.
```
All functions create parent dirs, return the output path, never show windows.

## Checkpoints — `checkpoints/`

`wm_v1.pt`, `wm_v2.pt`, `ppo_baseline.pt`, `dream_agent.pt` — each
`torch.save({"model": state_dict, "config": <relevant cfg dict>, "step": int})`.
Saved on CPU tensors.

## ONNX contract

`web/model_baseline.onnx`, `web/model_dream.onnx`. Input `frames` float32
[1,4,64,64] in [0,1] (stack order oldest→newest); output `logits` float32 [1,3];
opset 17. Demo uses argmax. Parity gate: max|torch−ort| < 1e-4 on 64 random inputs.

## Web demo — `web/`

- `constants.js` is GENERATED by `scripts/export_constants.py` (env-builder):
  `const C = {...config.ENV..., FPS: 30};` — web code reads only from `C`, no magic numbers.
- Simulation in JS mirrors the physics contract above verbatim, including
  `Math.floor(x + 0.5)` rounding; game state rendered into a `Uint8Array(64*64)`
  (0/255) identical to Python frames, kept as a 4-deep history, normalized /255
  into Float32Array for onnxruntime-web (CDN build). Display = separate scaled
  canvas with `imageSmoothingEnabled=false`; score shown in DOM, NOT in the 64px frame.
- Human = LEFT paddle (speed C.agent_speed for fairness), keys W/S or ↑/↓.
  Agent = RIGHT paddle via ONNX argmax each tick, model dropdown baseline/dream.
- If a model file fails to load: fall back to scripted opponent + visible banner.
  Serve `python -m http.server` from web/ (file:// blocks fetch of .onnx).

## Modal — `infra/`

- `infra/modal_common.py` (main-thread seeded): app "worldmodel-pong", volume
  "worldmodel-vol" at /vol, image = debian_slim py3.11 + pip deps + repo at /root
  (matches Modal's runner cwd/sys.path so `import config`, `import infra` resolve).
- modal-infra adds `infra/remote.py`: `run_module(module: str, argv: list[str])`
  Modal function (T4, volume, timeout from config.CAPS) that chdirs to /root,
  sets WM_ROOT=/vol, runs the module CLI via runpy, commits the volume; plus
  `launch(module, args, timeout_s, detach=False)` local helper invoking it, and
  `run_anywhere(module, args, scale_fallback)` — tries Modal, on ANY failure runs
  locally via subprocess at the fallback scale and logs loudly to FAILURES.md.
- Artifact retrieval: `modal volume get worldmodel-vol <remote> <local>` wrapper
  `fetch(remote_path, local_path)`.

## Appendix — agent clarifications (append below, own heading only)
