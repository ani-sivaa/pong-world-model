# DECISIONS.md — morning briefing

Running log of every design choice, fallback, and compromise, written as the run
progresses. Newest entries at the bottom of each section.

## Preflight & environment

- **Local Python is 3.13.2, not 3.11.** No 3.11 interpreter exists on this
  machine and installing one unattended (pyenv/brew) is slower and riskier than
  using 3.13, which torch fully supports. The Modal image pins 3.11 as
  specified, so the GPU path matches the spec exactly. All local code is
  version-agnostic (3.11+).
- **Venv at `.venv/`** (python3.13) rather than --user installs — avoids the
  PATH problems that bit the `modal` CLI install yesterday.
- **GPU = T4.** The models here are tiny (≤6M params, 64×64 frames); T4 is the
  cheapest Modal GPU and is more than adequate. Cost discipline over speed.
- **git init'd the repo** (it wasn't one). Rationale: unattended multi-agent
  overnight run + "a later failure must never destroy earlier results" — commits
  at phase boundaries are the standard mechanism for that guarantee.

## Deviations from the requested plan (all logged, none load-bearing)

- **INTERFACES.md written by the main thread before fanout**, not by eval-utils.
  The plan launches wave-1 agents in parallel, so env-builder/web-demo can't wait
  on eval-utils to publish contracts. Fixed contracts up front; agents append
  clarifications under their own heading only.
- **Subagents write `DECISIONS.<agent>.md`**, merged into this file by the main
  thread at phase boundaries. Four agents Edit-ing one file concurrently is a
  lost-update risk; separate files + merge is safe.

## Environment design (Phase 1 spec, decided up front)

- **Score is NOT rendered into the 64×64 frame.** Episodes end when a point is
  scored, so the score never changes within an episode — rendering it would add
  zero information for the agent/world model while adding a pixel-parity risk
  surface for the web demo. Score lives in the DOM in the demo.
- **Episode = one point** (or 500-step truncation). Keeps the done head
  meaningful and dream rollouts well-scoped.
- **Reward shaping: +0.1 per agent paddle hit** on top of ±1 for points. A dream
  horizon of ~40 steps rarely spans a full point, so pure ±1 reward would starve
  the dream agent of signal. All *evaluations* report raw points separately so
  the headline numbers are unshaped.
- **Serve is the only stochasticity** (seeded direction + vy at reset). A
  deterministic-given-state env is what a deterministic world model can learn;
  mid-episode randomness would force blurry predictions.
- **Opponent tracks at speed 1.0 < max ball vy 1.25** — a perfect tracker would
  never miss and no points would ever be scored; angled shots must be able to
  beat it so rewards exist.
- **Rounding convention `floor(x+0.5)`** in both Python and JS. Python `round()`
  banker-rounds (round(0.5)=0), JS `Math.round(0.5)=1` — this convention is
  bit-identical in both and is part of the pixel-parity contract.
- **1M transitions collected locally (deliverable) AND regenerated on Modal**
  from the same seed at training start (~minutes, vectorized env) instead of
  uploading ~4.1GB to the volume. Same code + same seed ⇒ identical data.

## Wave 1

(sections merged from subagent decision files as they complete)

## Phase 2 — world model

(main thread, written during the run)

## Phase 3

(written during the run)
