"""Geometry-grounded reward oracle for dream training (iteration-3 lever).

Motivation (see RESULTS.md / DECISIONS.md iterations 1-2): the v1 transfer gap
was traced to the world model's *learned* reward/done heads. With only ~53
scoring events per 1M transitions the WM never learned what a MISS looks like:
inside dreams the ball silently vanishes when it passes a paddle, the done head
never fires, and the reward head hallucinates free +0.1 "hits" in ball-less
frames. Iteration 2's ``--ball-guard`` patched the symptom (a ball-less frame
pays nothing and ends) but still trusts the reward head for every other step.

This module implements the DEFERRED lever from DECISIONS.md ("geometric reward
computed from decoded frames"): it ignores the learned heads entirely and
derives an HONEST per-step reward + termination signal from *decoded* frame
geometry, using the exact plane/paddle constants the real env uses
(config.ENV). Because a miss is read off the pixels rather than predicted by a
head that never saw one, the exploit the v1 agent farmed cannot pay off — no
matter how miscalibrated the WM's heads are.

An event is a *plane crossing*, which needs the ball's direction of travel, so
the signal is defined on THREE consecutive frames (prev -> cur -> nxt), each a
sigmoid probability map [B,1,64,64] in [0,1] (the representation dream rollout
feeds back into the WM). ``prev,cur`` come from the frame stack; ``nxt`` is the
WM's freshly dreamed frame. Validated per-rally against the true env reward in
``scripts/validate_geo_reward.py``.
"""
import torch

import config

_E = config.ENV
_W, _H = _E["W"], _E["H"]
_R_HIT = _E["r_hit"]
_R_CONCEDE = _E["r_concede"]
_R_SCORE = _E["r_score"]

# playfield columns strictly BETWEEN the two paddles: [left_x+paddle_w, right_x)
_PF_X0 = _E["left_x"] + _E["paddle_w"]        # 4  (first col past the left paddle)
_PF_X1 = _E["right_x"]                         # 60 (right plane; ball right edge bounces here)
_PF_Y0, _PF_Y1 = 1, _H - 1                     # rows between the two walls
_NEAR_R = _PF_X1 - 1 - 3.0                       # ball centroid "near" the right plane
_NEAR_L = _PF_X0 + 3.0                           # ... and the left plane


def _ball_stats(frames: torch.Tensor):
    """frames [B,1,64,64] probs -> (present[B] bool, cx[B], cy[B]) ball centroid
    over the playfield (walls + both paddle columns excluded)."""
    b = (frames[:, 0] > 0.5).float()
    play = b[:, _PF_Y0:_PF_Y1, _PF_X0:_PF_X1]
    counts = play.sum(dim=(1, 2))
    present = counts >= 1
    dev = frames.device
    ys = torch.arange(_PF_Y0, _PF_Y1, device=dev, dtype=torch.float32)
    xs = torch.arange(_PF_X0, _PF_X1, device=dev, dtype=torch.float32)
    denom = counts.clamp(min=1)
    cy = (play.sum(dim=2) * ys).sum(dim=1) / denom
    cx = (play.sum(dim=1) * xs).sum(dim=1) / denom
    return present, cx, cy


@torch.no_grad()
def geo_signal(prev: torch.Tensor, cur: torch.Tensor, nxt: torch.Tensor):
    """Honest (reward[B], cont[B]) read from decoded geometry over prev->cur->nxt.

    cont = 1.0 while the rally continues, 0.0 on a terminal (score/concede, or a
    ball that is simply gone). Events fire on the cur->nxt transition, gated by
    the direction of travel measured over prev->cur (a real crossing, not mere
    proximity), so each rally-ending event fires once.

    Events (agent = RIGHT paddle):
      * agent HIT  (+r_hit):    ball was heading right near the right plane and
        bounced back (paddle covered it) — rally continues.
      * agent MISS (r_concede): ball was heading right near the right plane and
        left the playfield with the paddle NOT covering it — rally ends.
      * agent SCORE (+r_score): symmetric event at the LEFT plane (opponent
        failed to cover) — rally ends.
      * ball otherwise gone:    reward 0, rally ends (same terminal semantics as
        --ball-guard: a dream without a ball is over).
    """
    B = prev.shape[0]
    dev = prev.device
    pv_present, pv_cx, _ = _ball_stats(prev)
    cur_present, cur_cx, _ = _ball_stats(cur)
    nx_present, nx_cx, _ = _ball_stats(nxt)

    reward = torch.zeros(B, device=dev)
    cont = torch.ones(B, device=dev)

    moving_right = pv_present & cur_present & (cur_cx > pv_cx + 0.25)
    moving_left = pv_present & cur_present & (cur_cx < pv_cx - 0.25)
    gone = ~nx_present
    bounced_left = nx_present & (nx_cx < cur_cx - 0.25)      # reversed to leftward
    bounced_right = nx_present & (nx_cx > cur_cx + 0.25)     # reversed to rightward

    # Physics IS the discriminator: a ball approaching a plane either bounces
    # back (the paddle covered it -> HIT / defended) or leaves the field (a
    # score against that side). Paddle alignment is implicit in whether the WM
    # bounces the ball, so we read the outcome rather than re-deriving coverage.

    # ---- RIGHT plane (agent side) ------------------------------------------
    approach_r = moving_right & (cur_cx >= _NEAR_R)
    hit_r = approach_r & bounced_left                       # agent returned it
    miss_r = approach_r & gone                              # ball got past agent
    reward = reward + _R_HIT * hit_r.float()
    reward = reward + _R_CONCEDE * miss_r.float()

    # ---- LEFT plane (opponent side) ----------------------------------------
    approach_l = moving_left & (cur_cx <= _NEAR_L)
    score_l = approach_l & gone                             # ball got past opponent
    reward = reward + _R_SCORE * score_l.float()

    # ---- termination -------------------------------------------------------
    dead = ~cur_present & ~nx_present
    unresolved_gone = cur_present & gone & ~miss_r & ~score_l
    end = miss_r | score_l | dead | unresolved_gone
    cont = torch.where(end, torch.zeros_like(cont), cont)
    return reward, cont
