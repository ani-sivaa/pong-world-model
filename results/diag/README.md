# Diagnostic probes — why the v1 dream agent failed on real Pong

Four independent probes, run in parallel, each testing one hypothesis with a
real experiment. Every probe writes a `*_probe.py` (the experiment, rerunnable)
and a `*_results.json` (the numbers). Summary of what each found:

| probe | files | hypothesis | verdict |
|---|---|---|---|
| Reward hallucination | `rewardhal_*` | reward head pays imagined hits in states the dream policy visits | **CONFIRMED** — head is near-perfect on-distribution (precision .958/recall .979); off-distribution the ball vanishes from dreams and 87% of reward events fire in ball-less frames; vanished-ball dreams pay 12× more |
| State coverage | `paddlecov_*` | WM error explodes in paddle positions the data never covered | **REFUTED, better finding** — paddle-y coverage is fine; the hole is *event* coverage: 53 scoring events in 1M transitions. Forced-miss lockstep: ball vanishes 20/20, done never fires (p≤.0016), WM says +0.10 where reality says −1.0 |
| Horizon compounding | `horizon_*` | H=40 dreams are too long; late reward is fiction | **REFUTED** — under dream-policy actions the dream is off-manifold from step 1 (8.8× honest drift at j=1); truncation is not the lever, action/state distribution is. Includes tracker-action control isolating the serve-init artifact |
| Policy degeneracy | `agentdegen_*` | the agent underconverged or learned a parked/degenerate policy | **CONFIRMED with nuance** — fully converged on its imagined objective; learned up-camping ball-dodging (84% UP, tracks neither real nor dreamed balls). The objective was corrupted, not the optimizer |

**Unified diagnosis:** the training data almost never showed the ball passing
the right paddle, so the world model renders misses as the ball silently
vanishing — no `done`, no −1 — and the reward head extrapolates free +0.1 hits
in those ball-less states. Dodging the ball was therefore the *optimal* policy
inside the dream. The agent found the exploit and converged on it.

**Fixes derived from these probes** (iteration 2):
1. `scripts/collect_mixed.py` — event-dense data: tracker 40% / PPO baseline
   35% / random 15% / v1 dream agent 10% (~100× more miss/score events).
2. WM v3 retrained from scratch on the mixed data (+ reward/done heads).
3. `agents/train_dream.py --ball-guard` — reward and continuation zeroed in
   ball-less dreamed frames: a dream without a ball is over, so vanish-states
   pay nothing regardless of residual WM flaws.
4. Horizon kept at 40 (probe 3: truncation destroys signal without adding honesty).

See RESULTS.md ("Iteration 2") for the full narrative and, once the retrain
lands, the before/after transfer numbers.
