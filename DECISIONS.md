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

- **Preflight caught two Modal infra bugs** (this is exactly why it exists):
  (1) `add_local_dir` ignore patterns used only `".venv/**"` — dockerignore
  semantics need the bare dir name too, so the first smoke run silently hashed/
  uploaded the multi-GB venv for 20+ minutes; killed, fixed with both forms.
  (2) repo mounted at `/root/proj` but Modal's runner resolves imports from
  `/root` → `ModuleNotFoundError: infra`; remounted at `/root`. Second run:
  `MODAL_SMOKE_OK` on a Tesla T4 in ~2 min.

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

## Phase 2 — world model (design, main thread)

- **U-Net skips from encoder to decoder.** Static content (walls, mostly-still
  paddles) rides through skips nearly free, so bottleneck capacity goes to
  dynamics. The classic risk (model learns identity) is countered by the
  change-weighted loss below.
- **BCE loss on logits, not MSE.** Pong pixels are near-binary; BCE keeps
  predictions crisp where MSE regresses to gray.
- **Change-weighted loss (w=15).** The 2×2 ball is ~0.1% of pixels; with
  uniform loss it blurs out of existence within a few rollout steps (the classic
  failure of this exact experiment). Per-pixel weight `1 + 15·|Δframe|`
  computed on *ground-truth* frames concentrates loss on the ball and moving
  paddle edges.
- **Multi-step unroll training (k=5) with gradient THROUGH the unroll,** feeding
  back sigmoid probabilities — the same convention the dream rollout uses, so
  train and inference distributions match. 35% of steps run at k=1 first
  (cheap warmup), then k=5. This is the single biggest lever for long-horizon
  coherence.
- **Bug caught in self-review before any training:** frame loss at a step where
  `done` fires would have trained the model to predict the *reset* frame
  (serve direction is RNG — unlearnable noise). Frame loss now masks
  `alive · (1-done)`; reward/done heads keep the `alive` mask so the done head
  still sees positive examples.
- **Fixed val windows (seeded rng(0))** so drift curves are comparable across
  ablation runs and model versions.

## Phase 3 — agents (design, main thread)

- **3a = PPO** (clip, GAE), not DQN: fastest reliable route to a decent Pong
  policy, and its value head matches the dream agent's A2C value head so the
  "identical policy architecture" requirement holds exactly (same PolicyNet).
- **3b = REINFORCE + value baseline inside frozen WM dreams.** No gradients
  through dynamics (not needed for discrete actions), WM outputs detached,
  predicted rewards clamped to [-1.5, 1.5] to guard against reward-head
  blowups, soft continuation mask `1 - p(done)` both discounts returns and
  weights losses — robust to a miscalibrated done head.
- **Dream starts are real frame-stacks sampled from the dataset** — keeps
  dreams anchored in-distribution rather than compounding from a fixed start.
- **Subagent tool restriction is prompt-level** (agents instructed which tools
  to use + hard file-ownership rules) rather than harness-level: custom agent
  definitions written mid-session may not be picked up, and a failed agent
  launch would cost more than the soft restriction risks.
