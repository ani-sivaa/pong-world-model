"""Vectorized Pong environment. Pure numpy (NO torch). See INTERFACES.md.

All physics constants come from config.ENV — zero physics literals here.
Rounding convention everywhere: int(floor(x + 0.5)) (identical in JS as
Math.floor(x + 0.5); Python round() banker-rounds and must NOT be used).

Public per-env state (float64 arrays of shape [n], readable by callers such as
the collection loop and PPO):
    ball_x, ball_y   : ball TOP-LEFT position
    ball_vx, ball_vy : ball velocity
    left_y, right_y  : paddle TOP-LEFT y (left = scripted opponent, right = agent)
    t                : int64 step counter within the current episode
"""
import numpy as np

import config

_E = config.ENV
_W, _H = _E["W"], _E["H"]
_WHITE = 255                       # pixel value for walls / paddles / ball

# ------- derived geometry (structural, computed from config.ENV) -------------
_TOP = 1                            # first playable row (row 0 is the top wall)
_BALL_Y_MIN = _TOP                  # ball top-left y reflects at this plane
_BALL_Y_MAX = _H - 1 - _E["ball_size"]     # ... and at this one (row H-1 is wall)
_PAD_Y_MIN = _TOP                   # paddle top-left y clamp
_PAD_Y_MAX = _H - 1 - _E["paddle_h"]
_RIGHT_PLANE = float(_E["right_x"])                     # ball RIGHT edge reflects here
_LEFT_PLANE = float(_E["left_x"] + _E["paddle_w"])      # ball LEFT edge reflects here
_SERVE_BALL_X = (_W - _E["ball_size"]) / 2.0            # ball centered at serve
_SERVE_BALL_Y = (_H - _E["ball_size"]) / 2.0
_SERVE_PAD_Y = (_H - _E["paddle_h"]) / 2.0              # paddles centered at serve
_HALF_PH = _E["paddle_h"] / 2.0
_HALF_BS = _E["ball_size"] / 2.0
_OFF_RIGHT_X = float(_W - 1)        # ball fully off-screen when ball_x > this


def _round_px(a):
    """The shared rounding convention: floor(x + 0.5), as int64 array."""
    return np.floor(np.asarray(a) + 0.5).astype(np.int64)


class VecPong:
    """n_envs independent Pong games stepped with pure array ops (no per-env loop)."""

    def __init__(self, n_envs: int, seed: int):
        self.n_envs = n_envs
        self.rng = np.random.default_rng(seed)   # single generator for all serves
        self._vy_choices = np.asarray(_E["serve_vy_choices"], dtype=np.float64)
        n = n_envs
        self.ball_x = np.zeros(n, np.float64)
        self.ball_y = np.zeros(n, np.float64)
        self.ball_vx = np.zeros(n, np.float64)
        self.ball_vy = np.zeros(n, np.float64)
        self.left_y = np.zeros(n, np.float64)
        self.right_y = np.zeros(n, np.float64)
        self.t = np.zeros(n, np.int64)
        self._reset_mask(np.ones(n, bool))

    # ------------------------------------------------------------------ reset
    def _reset_mask(self, mask):
        """Serve state for envs where mask is True. RNG draws sized by mask.sum()
        so identical trajectories give identical RNG streams (determinism)."""
        k = int(mask.sum())
        if k == 0:
            return
        self.ball_x[mask] = _SERVE_BALL_X
        self.ball_y[mask] = _SERVE_BALL_Y
        sign = self.rng.integers(0, 2, size=k).astype(np.float64) * 2.0 - 1.0
        self.ball_vx[mask] = sign * _E["ball_vx"]
        self.ball_vy[mask] = self._vy_choices[self.rng.integers(0, len(self._vy_choices), size=k)]
        self.left_y[mask] = _SERVE_PAD_Y
        self.right_y[mask] = _SERVE_PAD_Y
        self.t[mask] = 0

    def reset(self) -> np.ndarray:
        """Reset ALL envs; returns uint8 frames [n, H, W]."""
        self._reset_mask(np.ones(self.n_envs, bool))
        return self._render()

    # ------------------------------------------------------------------- step
    def step(self, actions):
        """actions: int array [n] in {0=stay, 1=up, 2=down} (agent = RIGHT paddle).

        Returns (frames uint8 [n,H,W], rewards float32 [n], dones bool [n], info)
        with info = {"paddle_hit": bool [n], "point": int8 [n],
                     "terminal_frame": uint8 [n,H,W]}.
        Done envs auto-reset; their returned frame is the new episode's first
        frame, while info["terminal_frame"] holds the true final (pre-reset)
        frame; zeros for non-done envs.
        """
        actions = np.asarray(actions).reshape(self.n_envs)
        n = self.n_envs
        rewards = np.zeros(n, np.float64)

        # 1. paddles move (agent by action; opponent scripted tracker); clamp.
        dy = np.where(actions == 1, -_E["agent_speed"], 0.0) \
           + np.where(actions == 2, _E["agent_speed"], 0.0)
        self.right_y = np.clip(self.right_y + dy, _PAD_Y_MIN, _PAD_Y_MAX)
        ball_cy = self.ball_y + _HALF_BS
        opp_diff = ball_cy - (self.left_y + _HALF_PH)
        opp_move = np.where(np.abs(opp_diff) > _E["opp_deadzone"],
                            np.sign(opp_diff) * _E["opp_speed"], 0.0)
        self.left_y = np.clip(self.left_y + opp_move, _PAD_Y_MIN, _PAD_Y_MAX)

        # 2. ball moves (keep prev x for the crossing tests).
        prev_x = self.ball_x.copy()
        self.ball_x = self.ball_x + self.ball_vx
        self.ball_y = self.ball_y + self.ball_vy

        # 3. wall bounce: reflect y at the planes, flip vy.
        lo = self.ball_y < _BALL_Y_MIN
        self.ball_y[lo] = 2.0 * _BALL_Y_MIN - self.ball_y[lo]
        self.ball_vy[lo] = -self.ball_vy[lo]
        hi = self.ball_y > _BALL_Y_MAX
        self.ball_y[hi] = 2.0 * _BALL_Y_MAX - self.ball_y[hi]
        self.ball_vy[hi] = -self.ball_vy[hi]

        # 4. paddle collisions (crossing test vs PREV x, paddles at NEW y, ball
        #    at post-wall-bounce y). Masks use the pre-reflection x and are
        #    mutually exclusive (planes are far apart vs |vx|).
        bs = float(_E["ball_size"])
        ball_cy = self.ball_y + _HALF_BS
        # right (agent) paddle: ball right edge crosses plane moving right.
        prev_r = prev_x + bs
        new_r = self.ball_x + bs
        ovl_r = (self.ball_y + bs > self.right_y) & (self.ball_y < self.right_y + _E["paddle_h"])
        hit_r = (prev_r <= _RIGHT_PLANE) & (_RIGHT_PLANE < new_r) & ovl_r
        # left (opponent) paddle: ball left edge crosses plane moving left.
        ovl_l = (self.ball_y + bs > self.left_y) & (self.ball_y < self.left_y + _E["paddle_h"])
        hit_l = (self.ball_x < _LEFT_PLANE) & (_LEFT_PLANE <= prev_x) & ovl_l

        self.ball_x[hit_r] = 2.0 * _RIGHT_PLANE - 2.0 * bs - self.ball_x[hit_r]
        self.ball_vx[hit_r] = -self.ball_vx[hit_r]
        off_r = (ball_cy[hit_r] - (self.right_y[hit_r] + _HALF_PH)) / _HALF_PH
        self.ball_vy[hit_r] = np.clip(off_r, -1.0, 1.0) * _E["max_vy"]
        rewards[hit_r] += _E["r_hit"]

        self.ball_x[hit_l] = 2.0 * _LEFT_PLANE - self.ball_x[hit_l]
        self.ball_vx[hit_l] = -self.ball_vx[hit_l]
        off_l = (ball_cy[hit_l] - (self.left_y[hit_l] + _HALF_PH)) / _HALF_PH
        self.ball_vy[hit_l] = np.clip(off_l, -1.0, 1.0) * _E["max_vy"]

        # 5. scoring: ball fully off-screen.
        off_right = self.ball_x > _OFF_RIGHT_X            # agent missed
        off_left = self.ball_x + bs < 0.0                 # opponent missed
        point = np.zeros(n, np.int8)
        point[off_right] = -1
        point[off_left] = 1
        rewards[off_right] += _E["r_concede"]
        rewards[off_left] += _E["r_score"]

        # 6. time / truncation.
        self.t += 1
        dones = off_right | off_left | (self.t >= _E["max_steps"])

        # 7. render post-step, pre-reset; then auto-reset done envs.
        frames = self._render()
        terminal = np.zeros_like(frames)
        if dones.any():
            d_idx = np.nonzero(dones)[0]
            terminal[d_idx] = frames[d_idx]
            self._reset_mask(dones)
            frames[d_idx] = self._render(d_idx)

        info = {"paddle_hit": hit_r, "point": point, "terminal_frame": terminal}
        return frames, rewards.astype(np.float32), dones, info

    # ----------------------------------------------------------------- render
    def _render(self, idx=None) -> np.ndarray:
        """uint8 [len(idx), H, W] frames for the given env indices (all if None).
        Background 0; walls, paddles and ball at _WHITE. Ball clipped at edges
        (it can be partially visible on a terminal scoring frame)."""
        if idx is None:
            idx = np.arange(self.n_envs)
        m = len(idx)
        f = np.zeros((m, _H, _W), np.uint8)
        f[:, 0, :] = _WHITE
        f[:, _H - 1, :] = _WHITE
        rows = np.arange(_E["paddle_h"])[None, :]
        ar = np.arange(m)[:, None]
        ly = _round_px(self.left_y[idx])[:, None] + rows
        ry = _round_px(self.right_y[idx])[:, None] + rows
        for c in range(_E["left_x"], _E["left_x"] + _E["paddle_w"]):
            f[ar, ly, c] = _WHITE
        for c in range(_E["right_x"], _E["right_x"] + _E["paddle_w"]):
            f[ar, ry, c] = _WHITE
        bx = _round_px(self.ball_x[idx])
        by = _round_px(self.ball_y[idx])
        ei = np.arange(m)
        for dyp in range(_E["ball_size"]):
            for dxp in range(_E["ball_size"]):
                r = by + dyp
                c = bx + dxp
                ok = (c >= 0) & (c < _W) & (r >= 0) & (r < _H)
                f[ei[ok], r[ok], c[ok]] = _WHITE
        return f


class Pong:
    """Thin n=1 wrapper around VecPong with unbatched returns."""

    def __init__(self, seed: int = config.SEED):
        self._vec = VecPong(1, seed)

    def reset(self):
        return self._vec.reset()[0]

    def step(self, action: int):
        frames, rewards, dones, info = self._vec.step(np.asarray([action]))
        info1 = {
            "paddle_hit": bool(info["paddle_hit"][0]),
            "point": int(info["point"][0]),
            "terminal_frame": info["terminal_frame"][0],
        }
        return frames[0], float(rewards[0]), bool(dones[0]), info1


def scripted_action(agent_y, ball_y, eps_random: float, rng: np.random.Generator):
    """Collection policy for the agent (RIGHT) paddle: eps-random tracker.

    agent_y, ball_y: top-left y positions (float array [n] or scalars).
    With prob eps_random per env: uniform random action; else track the ball
    center with deadzone config.ENV["opp_deadzone"]:
        paddle_cy < ball_cy - dz -> 2 (down); paddle_cy > ball_cy + dz -> 1 (up);
        else 0. Returns int64 array [n]. RNG draws are always full-batch so the
    call count is trajectory-independent (determinism).
    """
    agent_y = np.atleast_1d(np.asarray(agent_y, dtype=np.float64))
    ball_y = np.atleast_1d(np.asarray(ball_y, dtype=np.float64))
    pc = agent_y + _HALF_PH
    bc = ball_y + _HALF_BS
    dz = _E["opp_deadzone"]
    act = np.zeros(agent_y.shape[0], np.int64)
    act[pc < bc - dz] = 2
    act[pc > bc + dz] = 1
    rand_mask = rng.random(agent_y.shape[0]) < eps_random
    rand_act = rng.integers(0, config.N_ACTIONS, size=agent_y.shape[0])
    act[rand_mask] = rand_act[rand_mask]
    return act


# ============================ self-check + sample GIFs =======================
def _selfcheck():
    E = _E
    print("== VecPong self-check ==")

    # --- 1. wall bounce preserves |vy| and reflects y about the plane -------
    env = VecPong(1, seed=0)
    env.reset()
    env.ball_x[:] = 20.0
    env.ball_vx[:] = E["ball_vx"]
    env.ball_y[:] = 1.2
    env.ball_vy[:] = -1.0
    env.step(np.asarray([0]))
    assert env.ball_vy[0] == 1.0, f"top bounce vy {env.ball_vy[0]}"
    assert abs(env.ball_y[0] - (2 * _BALL_Y_MIN - (1.2 - 1.0))) < 1e-12, env.ball_y[0]
    env.ball_y[:] = float(_BALL_Y_MAX) - 0.3
    env.ball_vy[:] = 1.0
    env.step(np.asarray([0]))
    assert env.ball_vy[0] == -1.0, f"bottom bounce vy {env.ball_vy[0]}"
    assert abs(env.ball_y[0] - (2 * _BALL_Y_MAX - (_BALL_Y_MAX + 0.7))) < 1e-12
    print("wall bounce: OK (|vy| preserved, y reflected)")

    # --- 2. paddle hits flip vx and set vy by offset ------------------------
    env = VecPong(1, seed=0)
    env.reset()                                # paddles at 26 (centered)
    env.ball_x[:] = 57.0                       # right edge 59 -> 60.5 crosses 60
    env.ball_vx[:] = E["ball_vx"]
    env.ball_y[:] = 28.0
    env.ball_vy[:] = 0.0
    _, r, d, info = env.step(np.asarray([0]))
    exp_vy = np.clip((29.0 - (_SERVE_PAD_Y + _HALF_PH)) / _HALF_PH, -1, 1) * E["max_vy"]
    assert env.ball_vx[0] == -E["ball_vx"], "right hit must flip vx"
    assert abs(env.ball_vy[0] - exp_vy) < 1e-12, (env.ball_vy[0], exp_vy)
    assert abs(env.ball_x[0] - (2 * _RIGHT_PLANE - 2 * E["ball_size"] - 58.5)) < 1e-12
    assert info["paddle_hit"][0] and abs(r[0] - E["r_hit"]) < 1e-6, "agent hit reward"
    # left paddle
    env.ball_x[:] = 4.5                        # left edge 4.5 -> 3.0 crosses 4
    env.ball_vx[:] = -E["ball_vx"]
    env.ball_y[:] = 30.0
    env.ball_vy[:] = 0.0
    env.left_y[:] = _SERVE_PAD_Y
    _, r, d, info = env.step(np.asarray([0]))
    assert env.ball_vx[0] == E["ball_vx"], "left hit must flip vx"
    assert not info["paddle_hit"][0] and r[0] == 0.0, "opponent hit gives no reward"
    print("paddle hits: OK (vx flipped, vy from offset, r_hit on agent only)")

    # --- 3a. scoring is possible on BOTH sides — deterministic proof --------
    # (contract-exact trackers make ORGANIC points rare — ~1 per 20-50k
    #  scripted transitions, see DECISIONS.env-builder.md — so the mechanism is
    #  proven with engineered max-angle shots and organic scoring is asserted
    #  in a long soak below; check strengthened, not relaxed.)
    env = VecPong(1, seed=0)
    env.reset()
    env.ball_x[:] = 56.0                       # max-vy shot the opponent
    env.ball_vx[:] = -E["ball_vx"]             # (speed 1.0 < max_vy) cannot
    env.ball_y[:] = 16.0                       # reach across a full runway
    env.ball_vy[:] = E["max_vy"]
    env.left_y[:] = 11.0
    for k in range(60):
        f, r, d, info = env.step(np.asarray([0]))
        if d[0]:
            break
    assert d[0] and info["point"][0] == 1 and abs(r[0] - E["r_score"]) < 1e-6, \
        "opponent must miss a full-runway max-angle shot"
    assert info["terminal_frame"][0].any(), "terminal_frame must hold the final frame"
    assert not np.array_equal(f[0], info["terminal_frame"][0]), \
        "returned frame for a done env must be the fresh-serve frame"
    env2 = VecPong(1, seed=0)
    env2.reset()
    env2.ball_x[:] = 6.0                       # symmetric shot past a held-still
    env2.ball_vx[:] = E["ball_vx"]             # agent paddle
    env2.ball_y[:] = 16.0
    env2.ball_vy[:] = E["max_vy"]
    env2.right_y[:] = 11.0
    for k in range(60):
        f, r, d, info = env2.step(np.asarray([0]))
        if d[0]:
            break
    assert d[0] and info["point"][0] == -1 and abs(r[0] - E["r_concede"]) < 1e-6, \
        "agent must concede when it cannot reach the ball"
    print("scoring mechanism: OK (engineered max-angle shots score on both sides)")

    # --- 3b/4/6. scripted rollout: episodes end, frames binary, ep length ---
    n = 16
    env = VecPong(n, seed=config.SEED)
    rng = np.random.default_rng(config.SEED)
    env.reset()
    plus = minus = ndone = 0
    ep_steps = np.zeros(n, np.int64)
    ep_lengths = []
    binary_ok = True
    for _ in range(2000):
        a = scripted_action(env.right_y, env.ball_y, config.COLLECT["eps_random"], rng)
        f, r, d, info = env.step(a)
        plus += int((info["point"] == 1).sum())
        minus += int((info["point"] == -1).sum())
        ep_steps += 1
        if d.any():
            ep_lengths.extend(ep_steps[d].tolist())
            ep_steps[d] = 0
            ndone += int(d.sum())
        u = np.unique(f)
        if not np.all(np.isin(u, [0, 255])):
            binary_ok = False
    assert ndone > 0, "no episode ever ended"
    assert binary_ok, "frames must contain only {0, 255}"
    mean_len = float(np.mean(ep_lengths))
    print(f"scripted 2000 steps x {n} envs: +points={plus} -points={minus} "
          f"episodes={ndone} mean_ep_len={mean_len:.1f} frames binary: OK")

    # --- 3c. organic scoring on BOTH sides under the collection policy ------
    n = 64
    env = VecPong(n, seed=config.SEED)
    rng = np.random.default_rng(config.SEED)
    env.reset()
    soak_steps = 31_250                        # 2M transitions
    plus = minus = 0
    for _ in range(soak_steps):
        a = scripted_action(env.right_y, env.ball_y, config.COLLECT["eps_random"], rng)
        _, _, _, info = env.step(a)
        plus += int((info["point"] == 1).sum())
        minus += int((info["point"] == -1).sum())
    total = soak_steps * n
    assert plus > 0, f"no organic agent point in {total} scripted transitions"
    assert minus > 0, f"no organic opponent point in {total} scripted transitions"
    print(f"organic scoring soak ({total:,} transitions): +points={plus} "
          f"-points={minus}: OK")

    # --- 5. determinism ------------------------------------------------------
    def run(seed_env, seed_pol, steps):
        e = VecPong(4, seed_env)
        g = np.random.default_rng(seed_pol)
        e.reset()
        out = []
        for _ in range(steps):
            a = scripted_action(e.right_y, e.ball_y, config.COLLECT["eps_random"], g)
            f, _, _, _ = e.step(a)
            out.append(f)
        return np.stack(out)

    fa = run(7, 123, 500)
    fb = run(7, 123, 500)
    assert fa.shape == (500, 4, _H, _W) and np.array_equal(fa, fb), "determinism failed"
    print("determinism: OK (two VecPong(4, seed=7) runs bit-identical over 500 steps)")
    print("== all self-checks passed ==")


def _sample_gifs():
    import imageio.v2 as imageio
    out_dir = config.RESULTS_DIR / "env_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed, steps in [(1, 300), (2, 300), (3, 400)]:
        env = VecPong(1, seed)
        rng = np.random.default_rng(seed)
        frames = [env.reset()[0]]
        for _ in range(steps):
            a = scripted_action(env.right_y, env.ball_y, config.COLLECT["eps_random"], rng)
            f, _, _, _ = env.step(a)
            frames.append(f[0])
        arr = np.stack(frames)                                   # [T, 64, 64]
        up = np.kron(arr, np.ones((1, 4, 4), np.uint8))          # 4x NN upscale
        path = out_dir / f"scripted_seed{seed}_{steps}steps.gif"
        try:
            imageio.mimsave(path, list(up), fps=30, loop=0)
        except TypeError:                       # newer imageio: duration in ms
            imageio.mimsave(path, list(up), duration=1000.0 / 30.0, loop=0)
        print(f"wrote {path}")


if __name__ == "__main__":
    _selfcheck()
    _sample_gifs()
