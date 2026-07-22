# DECISIONS — web-demo

Judgment calls made while building `web/index.html`, `web/pong.js`,
`web/README.md`. Written while `pong/env.py`, `web/constants.js`, and the
`.onnx` files did not exist yet (parallel agents), so everything is coded
strictly against INTERFACES.md + config.py.

## RNG parity stance
The contract's seeded-RNG determinism matters for training-data
reproducibility, not for the demo. `pong.js` uses `Math.random()` for the two
serve choices (vx sign, vy from `C.serve_vy_choices`) — no attempt to replicate
numpy's PCG64 stream in JS. What IS replicated exactly is the *dynamics*: float
positions, `Math.floor(x + 0.5)` rounding, the verbatim 7-stage step order,
crossing tests vs prev x, reflect formulas, clamps, and scoring thresholds. Any
(serve state, action sequence) pair produces the identical trajectory in Python
and JS.

## Crossing-test mirroring (Phase 4 should verify vs env.py)
The contract writes the crossing test in rightward form:
`prev_edge <= plane < new_edge` (right paddle, ball right edge vs x=60).
For the left paddle I mirrored the strictness under x -> -x, giving
`new_left < plane <= prev_left` (ball left edge vs x=4). If env.py placed the
strict/inclusive bounds differently on the left plane, parity breaks only in a
sub-pixel edge case, but Phase 4 should diff a scripted-vs-scripted trajectory
(same serve, same actions) between Python and a JS port run to confirm.
Reflect formulas used: right — ball right edge about the plane
(`bx' = 2*60 - bx - 2*ball_size`); left — ball left edge (`bx' = 2*4 - bx`).

## Truncation in an endless match
The env terminates on `t >= max_steps` (truncation). The demo keeps the same
counter and, when it fires, re-serves WITHOUT changing the score — dynamics
identical to an episode boundary, but the match never ends. Points likewise
just update the DOM scoreboard and re-serve.

## Frame stack at episode boundaries
On every serve (match start, after a point, after truncation) the history is
refilled with the new serve frame repeated 4x — same as a fresh episode's
initial observation in training (and per the "repeat the first frame" rule for
match start). Mid-rally, frames shift oldest->newest. The tensor is
`Float32Array(1*4*64*64)` = frames/255, oldest first.

## FRAME_STACK is not in `C`
`config.FRAME_STACK` lives outside `config.ENV`, and `constants.js` exports
only `{...ENV, FPS}`. The stack depth 4 is also fixed by the ONNX contract
(input `[1,4,64,64]`), so pong.js uses `C.FRAME_STACK ?? 4` — reads it if the
exporter ever adds it, otherwise falls back to the contract value. This is the
only "number" in pong.js not sourced from `C`, besides structural facts (wall =
rows 0 and H-1 with thickness 1) and the pure display scale (8x).

## Timing: awaited async inference inside a fixed-timestep accumulator
`session.run()` is async. Rather than acting on a one-frame-stale result, the
game loop awaits inference inside the step pump, so the agent always acts on
the CURRENT 4-frame stack — exactly like training. Safety: the pump is
re-entrancy-guarded (rAF never overlaps two pumps) and the accumulator is
clamped to 5 frames of backlog, so a slow model slows the game gracefully
instead of triggering a catch-up death spiral. Latency is measured around
`session.run` and shown as a rolling mean of the last 120 inferences.

## Fallback ladder (game is always playable)
- `typeof C === "undefined"` (constants.js missing): red banner
  "constants.js missing — run scripts/export_constants.py", hard halt.
- `typeof ort === "undefined"` (CDN blocked/offline), model fetch or
  `InferenceSession.create` failure (e.g. .onnx not exported yet), or a runtime
  inference error: amber banner "model not loaded — scripted fallback"; the
  right paddle switches to the contract's scripted tracker at `C.opp_speed`
  with `C.opp_deadzone`. The ONNX agent moves at `C.agent_speed`; the human
  always moves at `C.agent_speed` (fairness, per the contract).
- Dropdown swap: the new session is created and swapped live; while loading
  (and if the load FAILS) the scripted fallback drives, per the spec — the old
  session is not kept. Re-selecting a model retries the load, so dropping the
  .onnx files in later "just works" after a dropdown change (or reload).
- A stale-load guard ignores a slow `create()` resolving after the user has
  already switched models again.

## Input details
- Hold-to-move: keydown/keyup tracked as booleans; W/S and ArrowUp/ArrowDown
  (arrows preventDefault to stop page scroll). Up and down both held = stay
  (action 0), matching "no net input".
- Touch buttons use pointer events with pointerup/cancel/leave all clearing the
  hold, so paddles can't stick on mobile.

## Rendering
64x64 `Uint8Array` (values {0,255}, identical to Python frames) -> offscreen
64x64 canvas via `putImageData` -> `drawImage` onto the visible canvas at 8x
with `imageSmoothingEnabled = false`. The visible frame is redrawn every rAF;
physics only advances on accumulator ticks at `C.FPS`. Score, model status, and
latency live in the DOM — the 64px frame stays exactly what the model saw in
training.

## ONNX I/O names
Contract fixes input "frames" / output "logits"; pong.js uses
`session.inputNames[0]` / `out.logits ?? out[session.outputNames[0]]` as a
belt-and-suspenders read of the same names. Argmax runs over however many
logits come back (3 per contract) — no hardcoded action count.
