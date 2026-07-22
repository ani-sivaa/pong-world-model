"""Phase 3a — PPO baseline on REAL Pong. The guaranteed deliverable.

Vectorized numpy envs (CPU) + small CNN policy (GPU/MPS). Standard PPO-clip
with GAE; hyperparameters in config.PPO.

CLI: python -m agents.train_ppo --scale full [--out checkpoints/ppo_baseline.pt]
     [--max-seconds S] [--steps N]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from agents.policy import PolicyNet
from pong.env import VecPong


class StackedVecEnv:
    """VecPong + 4-frame stacking with correct auto-reset semantics:
    on done, the env returns the NEW episode's first frame -> restart the stack."""

    def __init__(self, n, seed):
        self.env = VecPong(n, seed=seed)
        self.n = n
        f = self.env.reset()                              # [n,64,64] u8
        self.stacks = np.repeat(f[:, None], config.FRAME_STACK, axis=1)

    def obs(self):
        return self.stacks

    def step(self, actions):
        f, r, d, info = self.env.step(actions)
        self.stacks = np.concatenate([self.stacks[:, 1:], f[:, None]], axis=1)
        if d.any():
            self.stacks[d] = f[d][:, None]                # fresh episode: repeat
        return r, d, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="local", choices=list(config.SCALES))
    ap.add_argument("--out", default=str(config.CKPT_DIR / "ppo_baseline.pt"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-seconds", type=int, default=config.CAPS["ppo"])
    args = ap.parse_args()

    config.seed_everything()
    dev = config.get_device()
    P = config.PPO
    total_steps = args.steps or config.SCALES[args.scale]["ppo_steps"]
    N, T = P["n_envs"], P["rollout"]
    n_iters = max(1, total_steps // (N * T))

    venv = StackedVecEnv(N, seed=config.SEED)
    policy = PolicyNet().to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=P["lr"], eps=1e-5)

    obs_buf = np.zeros((T, N, config.FRAME_STACK, 64, 64), dtype=np.uint8)
    act_buf = np.zeros((T, N), dtype=np.int64)
    logp_buf = np.zeros((T, N), dtype=np.float32)
    rew_buf = np.zeros((T, N), dtype=np.float32)
    done_buf = np.zeros((T, N), dtype=np.float32)
    val_buf = np.zeros((T, N), dtype=np.float32)

    ep_ret = np.zeros(N); ep_len = np.zeros(N, dtype=np.int64); ep_pts = np.zeros(N)
    stats = {"iter": [], "steps": [], "mean_point": [], "mean_shaped": [],
             "mean_len": [], "loss_pi": [], "loss_v": [], "entropy": []}
    recent = {"points": [], "shaped": [], "lens": []}
    t0 = time.time()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ppo] dev={dev.type} iters={n_iters} envs={N} rollout={T} "
          f"total={n_iters*N*T}", flush=True)

    for it in range(1, n_iters + 1):
        # ---------------- rollout collection --------------------------------
        for t in range(T):
            obs = venv.obs()
            obs_buf[t] = obs
            with torch.no_grad():
                x = torch.from_numpy(obs).to(dev).float() / 255.0
                logits, value = policy(x)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp_buf[t] = dist.log_prob(a).cpu().numpy()
                val_buf[t] = value.cpu().numpy()
            acts = a.cpu().numpy()
            act_buf[t] = acts
            r, d, info = venv.step(acts)
            rew_buf[t], done_buf[t] = r, d.astype(np.float32)
            ep_ret += r; ep_len += 1; ep_pts += info["point"]
            for i in np.flatnonzero(d):
                recent["shaped"].append(ep_ret[i]); recent["lens"].append(ep_len[i])
                recent["points"].append(ep_pts[i])
                ep_ret[i] = ep_len[i] = ep_pts[i] = 0

        # ---------------- GAE ------------------------------------------------
        with torch.no_grad():
            x = torch.from_numpy(venv.obs()).to(dev).float() / 255.0
            _, last_v = policy(x)
            last_v = last_v.cpu().numpy()
        adv = np.zeros_like(rew_buf)
        gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            nxt_v = last_v if t == T - 1 else val_buf[t + 1]
            nonterm = 1.0 - done_buf[t]
            delta = rew_buf[t] + P["gamma"] * nxt_v * nonterm - val_buf[t]
            gae = delta + P["gamma"] * P["gae_lambda"] * nonterm * gae
            adv[t] = gae
        ret = adv + val_buf

        b_obs = obs_buf.reshape(T * N, config.FRAME_STACK, 64, 64)
        b_act = torch.from_numpy(act_buf.reshape(-1)).to(dev)
        b_logp = torch.from_numpy(logp_buf.reshape(-1)).to(dev)
        b_adv = torch.from_numpy(adv.reshape(-1)).to(dev)
        b_ret = torch.from_numpy(ret.reshape(-1)).to(dev)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # ---------------- PPO update ----------------------------------------
        idx = np.arange(T * N)
        mb = T * N // P["minibatches"]
        for _ in range(P["epochs"]):
            np.random.shuffle(idx)
            for s in range(0, T * N, mb):
                j = idx[s:s + mb]
                x = torch.from_numpy(b_obs[j]).to(dev).float() / 255.0
                logits, v = policy(x)
                dist = torch.distributions.Categorical(logits=logits)
                jt = torch.from_numpy(j).to(dev)
                logp = dist.log_prob(b_act[jt])
                ratio = (logp - b_logp[jt]).exp()
                pg = -torch.min(
                    ratio * b_adv[jt],
                    ratio.clamp(1 - P["clip"], 1 + P["clip"]) * b_adv[jt]).mean()
                vloss = F.mse_loss(v, b_ret[jt])
                ent = dist.entropy().mean()
                loss = pg + P["vf_coef"] * vloss - P["ent_coef"] * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), P["grad_clip"])
                opt.step()

        if it % 5 == 0 or it == n_iters:
            mp = float(np.mean(recent["points"][-200:])) if recent["points"] else 0.0
            ms = float(np.mean(recent["shaped"][-200:])) if recent["shaped"] else 0.0
            ml = float(np.mean(recent["lens"][-200:])) if recent["lens"] else 0.0
            stats["iter"].append(it); stats["steps"].append(it * N * T)
            stats["mean_point"].append(mp); stats["mean_shaped"].append(ms)
            stats["mean_len"].append(ml); stats["loss_pi"].append(float(pg))
            stats["loss_v"].append(float(vloss)); stats["entropy"].append(float(ent))
            print(f"  iter {it}/{n_iters} steps={it*N*T} point={mp:+.3f} "
                  f"shaped={ms:+.3f} len={ml:.0f} ent={float(ent):.3f} "
                  f"{time.time()-t0:.0f}s", flush=True)
        if it % 20 == 0 or it == n_iters:
            torch.save({"model": {k: v.cpu() for k, v in policy.state_dict().items()},
                        "config": config.PPO, "step": it * N * T}, out)
        if time.time() - t0 > args.max_seconds:
            print(f"[ppo] WALL-CLOCK CAP at iter {it} — saving and exiting",
                  flush=True)
            break

    torch.save({"model": {k: v.cpu() for k, v in policy.state_dict().items()},
                "config": config.PPO, "step": it * N * T}, out)
    lp = config.RESULTS_DIR / "ppo_train_log.json"
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps(stats))
    try:
        from evalutils import plots
        plots.line_plot(stats["steps"],
                        {"mean point/ep": stats["mean_point"],
                         "mean shaped return": stats["mean_shaped"]},
                        title="PPO baseline on real Pong", xlabel="env steps",
                        ylabel="per-episode value",
                        out_path=config.RESULTS_DIR / "ppo_train.png")
    except Exception as e:
        print(f"[ppo] plot skipped: {e}")
    print(f"PPO_OK final_point={stats['mean_point'][-1] if stats['mean_point'] else 0:+.3f} "
          f"ckpt={out} wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
