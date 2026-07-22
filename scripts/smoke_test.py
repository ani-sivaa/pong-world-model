"""PREFLIGHT smoke test — end-to-end at tiny scale, fully self-contained.

Stages: (1) mini-Pong collect 500 transitions, (2) 30-second world-model train,
(3) 100-step REINFORCE agent train, (4) ONNX export + onnxruntime parity.
Exit code 0 iff every stage passes. Runs on CPU in ~2 minutes.

This intentionally inlines a simplified env/model — the real ones are built by
wave-1 agents AFTER this gate passes. It exists to catch import/credential/API
breakage before any expensive work starts.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def stage(name):
    print(f"\n=== SMOKE: {name} ===", flush=True)


# ---------------------------------------------------------------- mini env --
class MiniPong:
    """Simplified single-env Pong, same frame/action spec as the real one."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.E = config.ENV
        self.reset()

    def reset(self):
        E = self.E
        self.by, self.bx = E["H"] / 2 - 1, E["W"] / 2 - 1
        self.vx = E["ball_vx"] * (1 if self.rng.random() < 0.5 else -1)
        self.vy = float(self.rng.choice(E["serve_vy_choices"]))
        self.ly = self.ry = E["H"] / 2 - E["paddle_h"] / 2
        self.t = 0
        return self._render()

    def _render(self):
        E = self.E
        f = np.zeros((E["H"], E["W"]), dtype=np.uint8)
        f[0, :] = f[-1, :] = 255
        r = lambda v: int(np.floor(v + 0.5))
        f[r(self.ly):r(self.ly) + E["paddle_h"], E["left_x"]:E["left_x"] + 2] = 255
        f[r(self.ry):r(self.ry) + E["paddle_h"], E["right_x"]:E["right_x"] + 2] = 255
        f[r(self.by):r(self.by) + 2, r(self.bx):r(self.bx) + 2] = 255
        return f

    def step(self, a):
        E = self.E
        self.ry += (0, -E["agent_speed"], E["agent_speed"])[a]
        self.ry = float(np.clip(self.ry, 1, E["H"] - 1 - E["paddle_h"]))
        # crude opponent
        if self.ly + E["paddle_h"] / 2 < self.by:
            self.ly = min(self.ly + E["opp_speed"], E["H"] - 1 - E["paddle_h"])
        else:
            self.ly = max(self.ly - E["opp_speed"], 1)
        self.bx += self.vx
        self.by += self.vy
        if self.by < 1: self.by, self.vy = 2 - self.by, -self.vy
        if self.by > E["H"] - 3: self.by, self.vy = 2 * (E["H"] - 3) - self.by, -self.vy
        rew, done = 0.0, False
        if self.vx > 0 and self.bx + 2 > E["right_x"] and self.ry - 2 < self.by < self.ry + E["paddle_h"]:
            self.vx = -self.vx
            rew = E["r_hit"]
        if self.vx < 0 and self.bx < E["left_x"] + 2 and self.ly - 2 < self.by < self.ly + E["paddle_h"]:
            self.vx = -self.vx
        if self.bx > E["W"] - 1: rew, done = E["r_concede"], True
        if self.bx + 2 < 0: rew, done = E["r_score"], True
        self.t += 1
        if self.t >= E["max_steps"]: done = True
        f = self._render()
        if done: f = self.reset()
        return f, rew, done


def main():
    t_start = time.time()
    config.seed_everything()
    import torch
    import torch.nn as nn

    ok = {}

    # ---- stage 1: collect 500 transitions -------------------------------
    stage("collect 500 transitions")
    env = MiniPong(seed=config.SEED)
    frames, actions, rewards, dones = [env._render()], [], [], []
    rng = np.random.default_rng(config.SEED)
    for _ in range(500):
        a = int(rng.integers(3))
        f, r, d = env.step(a)
        frames.append(f); actions.append(a); rewards.append(r); dones.append(d)
    frames = np.stack(frames); actions = np.array(actions)
    assert frames.dtype == np.uint8 and frames.shape[1:] == (64, 64)
    assert frames[:-1].std() > 0, "frames are blank"
    print(f"collected {len(actions)} transitions, {sum(dones)} dones, "
          f"reward sum {sum(rewards):.1f}")
    ok["collect"] = True

    # ---- stage 2: 30-second world-model train ---------------------------
    stage("30s world-model train")
    dev = torch.device("cpu")  # smoke stays CPU: exercises the weakest path
    wm = nn.Sequential(
        nn.Conv2d(4, 32, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(32, 1, 4, 2, 1),
    ).to(dev)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    x_all = frames.astype(np.float32) / 255.0
    valid = [t for t in range(3, 499) if not any(dones[max(0, t - 3):t + 1])]
    t0, steps, first_loss, last_loss = time.time(), 0, None, None
    while time.time() - t0 < 30:
        idx = rng.choice(valid, size=16)
        stack = np.stack([x_all[i - 3:i + 1] for i in idx])
        target = np.stack([x_all[i + 1] for i in idx])[:, None]
        pred = wm(torch.from_numpy(stack).to(dev))
        loss = nn.functional.binary_cross_entropy_with_logits(
            pred, torch.from_numpy(target).to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
        if first_loss is None: first_loss = loss.item()
        last_loss = loss.item(); steps += 1
    print(f"{steps} steps in 30s, loss {first_loss:.4f} -> {last_loss:.4f}")
    assert last_loss < first_loss, "world-model loss did not decrease"
    ok["wm_train"] = True

    # ---- stage 3: 100-step agent train (REINFORCE) ----------------------
    stage("100-step agent train")
    policy = nn.Sequential(
        nn.Conv2d(4, 16, 8, 4), nn.ReLU(),
        nn.Conv2d(16, 32, 4, 2), nn.ReLU(),
        nn.Flatten(), nn.Linear(32 * 6 * 6, 64), nn.ReLU(), nn.Linear(64, 3),
    ).to(dev)
    popt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    env2 = MiniPong(seed=1)
    hist = [env2._render()] * 4
    for step in range(100):
        stack = torch.from_numpy(
            np.stack(hist)[None].astype(np.float32) / 255.0).to(dev)
        logits = policy(stack)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        f, r, d = env2.step(int(a))
        hist = hist[1:] + [f]
        loss = -dist.log_prob(a) * (r + 0.01)  # nonsense objective; mechanics only
        popt.zero_grad(); loss.backward(); popt.step()
    print("100 policy-gradient steps completed")
    ok["agent_train"] = True

    # ---- stage 4: ONNX export + parity ----------------------------------
    stage("ONNX export + onnxruntime parity")
    import onnxruntime as ort
    out = Path(config.RESULTS_DIR) / "smoke_policy.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 4, 64, 64)
    policy.eval()
    torch.onnx.export(
        policy, (dummy,), str(out), opset_version=config.ONNX["opset"],
        input_names=[config.ONNX["input_name"]],
        output_names=[config.ONNX["output_name"]], dynamo=False)
    sess = ort.InferenceSession(str(out))
    test = np.random.rand(1, 4, 64, 64).astype(np.float32)
    with torch.no_grad():
        torch_out = policy(torch.from_numpy(test)).numpy()
    ort_out = sess.run(None, {config.ONNX["input_name"]: test})[0]
    diff = float(np.abs(torch_out - ort_out).max())
    print(f"parity max|torch-ort| = {diff:.2e}")
    assert diff < 1e-4, "ONNX parity failed"
    ok["onnx"] = True

    print(f"\nSMOKE_LOCAL_OK all {len(ok)} stages passed "
          f"in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
