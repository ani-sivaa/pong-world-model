"""Phase 3b — train a policy ENTIRELY inside the world model's imagination.

The policy never touches real Pong during training. Dreams start from real
frame-stacks sampled from the dataset; the frozen v2 world model (frame +
reward + done heads) rolls forward under the policy's sampled actions.

Objective: actor-critic REINFORCE on imagined trajectories —
  discounted returns use the PREDICTED reward and a soft continuation mask
  (1 - predicted done prob); the value head bootstraps the horizon tail.
Gradients flow through the policy only; the world model is frozen and its
outputs are detached (REINFORCE needs no dynamics gradients).

CLI: python -m agents.train_dream --scale full --wm checkpoints/wm_v2.pt
     [--out checkpoints/dream_agent.pt] [--max-seconds S] [--updates N]
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
from agents.geo_reward import geo_signal
from wm.data import TransitionData
from wm.model import load_wm


def ball_alive(frames: torch.Tensor) -> torch.Tensor:
    """Ball-existence guard (diagnostic probe fix): frames [B,1,64,64] sigmoid
    probs -> float [B] 1.0 iff a ball is visibly present in the dreamed frame.

    Rationale: with miss-events nearly absent from v1 data, the WM lets the
    ball VANISH when it passes a paddle, the done head never fires, and the
    reward head hallucinates hits in ball-less frames — the exploit the v1
    dream agent farmed. Zeroing reward AND continuation when no ball pixels
    exist outside walls/paddle columns makes vanish-states worthless: a dream
    without a ball is over. (Legit dreamed scores also end here — correct.)
    """
    f = (frames[:, 0] > 0.5).float().clone()
    f[:, 0, :] = 0
    f[:, 63, :] = 0                    # walls
    f[:, :, 2:4] = 0
    f[:, :, 60:62] = 0                 # paddle columns
    return (f.sum(dim=(1, 2)) >= 2).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="local", choices=list(config.SCALES))
    ap.add_argument("--wm", default=str(config.CKPT_DIR / "wm_v2.pt"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=str(config.CKPT_DIR / "dream_agent.pt"))
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ball-guard", action="store_true",
                    help="zero reward+continuation in ball-less dreamed frames")
    ap.add_argument("--geo-reward", action="store_true",
                    help="derive reward+continuation from decoded frame GEOMETRY "
                         "(plane crossings) instead of the WM's learned heads; "
                         "an honest miss/hit oracle that the exploit can't game")
    ap.add_argument("--init-from", default=None,
                    help="policy checkpoint to continue training from")
    ap.add_argument("--max-seconds", type=int, default=config.CAPS["dream"])
    args = ap.parse_args()

    config.seed_everything()
    dev = config.get_device()
    D = config.DREAM
    sc = config.SCALES[args.scale]
    updates = args.updates or sc["dream_updates"]
    B = args.batch or D["batch"]
    H, gamma = args.horizon or D["horizon"], D["gamma"]
    data_dir = args.data or (config.DATA_DIR / args.scale)

    wm = load_wm(args.wm, dev, with_heads=True).eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    policy = PolicyNet().to(dev)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        policy.load_state_dict(ckpt["model"])
        print(f"[dream] continuing from {args.init_from} (update {ckpt['step']})",
              flush=True)
    lr = args.lr or D["lr"]
    opt = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    data = TransitionData(data_dir)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    log = {"update": [], "dream_return": [], "dream_len": [], "entropy": [],
           "loss_pi": [], "loss_v": []}
    t0 = time.time()
    print(f"[dream] dev={dev.type} updates={updates} B={B} H={H} wm={args.wm}",
          flush=True)

    for u in range(1, updates + 1):
        stacks_u8, *_ = data.sample(B, 1)                    # real starts
        stack = torch.from_numpy(stacks_u8.astype(np.float32) / 255.0).to(dev)

        logps, values, entropies, rewards, conts = [], [], [], [], []
        for _ in range(H):
            logits, v = policy(stack)                        # WITH grad
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logps.append(dist.log_prob(a))
            values.append(v)
            entropies.append(dist.entropy())
            with torch.no_grad():                            # frozen dynamics
                fl, r, dl = wm(stack, a)
                nxt = torch.sigmoid(fl)                      # [B,1,64,64]
                if args.geo_reward:
                    # honest reward/termination read off decoded geometry
                    # (prev, cur, nxt); ignores the WM's miscalibrated heads
                    g_rew, g_cont = geo_signal(stack[:, -2:-1], stack[:, -1:], nxt)
                    rewards.append(g_rew)
                    conts.append(g_cont)
                else:
                    ok = ball_alive(nxt) if args.ball_guard else 1.0
                    rewards.append(r.clamp(-1.5, 1.5) * ok)      # no ball -> no pay
                    conts.append((1.0 - torch.sigmoid(dl)) * ok) # no ball -> dream over
                stack = torch.cat([stack[:, 1:], nxt], dim=1)
        with torch.no_grad():
            _, boot = policy(stack)                          # tail bootstrap

        # discounted returns with soft done-masking, backwards
        R = boot
        returns = [None] * H
        for j in reversed(range(H)):
            R = rewards[j] + gamma * conts[j] * R
            returns[j] = R
        returns = torch.stack(returns)                       # [H,B]
        values_t = torch.stack(values)
        logps_t = torch.stack(logps)
        ent_t = torch.stack(entropies)
        # weight: probability the dream is still alive when the step happens
        alive = torch.cumprod(torch.cat(
            [torch.ones(1, B, device=dev), torch.stack(conts)[:-1]], 0), dim=0)

        adv = (returns - values_t).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss_pi = -(logps_t * adv * alive).mean()
        loss_v = (F.mse_loss(values_t, returns.detach(), reduction="none")
                  * alive).mean()
        loss = loss_pi + D["vf_coef"] * loss_v - D["ent_coef"] * (ent_t * alive).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), D["grad_clip"])
        opt.step()

        if u % 10 == 0 or u == 1:
            dream_ret = float((torch.stack(rewards) * alive).sum(0).mean())
            dream_len = float(alive.sum(0).mean())
            ent = float((ent_t * alive).sum(0).mean() / alive.sum(0).mean())
            log["update"].append(u); log["dream_return"].append(dream_ret)
            log["dream_len"].append(dream_len); log["entropy"].append(ent)
            log["loss_pi"].append(float(loss_pi)); log["loss_v"].append(float(loss_v))
            print(f"  update {u}/{updates} dream_return={dream_ret:+.3f} "
                  f"alive_len={dream_len:.1f}/{H} ent={ent:.3f} "
                  f"{time.time()-t0:.0f}s", flush=True)
        if u % 100 == 0 or u == updates:
            torch.save({"model": {k: v.cpu() for k, v in policy.state_dict().items()},
                        "config": dict(D, wm=str(args.wm)), "step": u}, out)
        if time.time() - t0 > args.max_seconds:
            print(f"[dream] WALL-CLOCK CAP at update {u} — saving and exiting",
                  flush=True)
            break

    torch.save({"model": {k: v.cpu() for k, v in policy.state_dict().items()},
                "config": dict(D, wm=str(args.wm)), "step": u}, out)
    lp = config.RESULTS_DIR / "dream_train_log.json"
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps(log))
    try:
        from evalutils import plots
        plots.line_plot(log["update"],
                        {"dream return (shaped)": log["dream_return"]},
                        title="Dream agent — return inside the world model",
                        xlabel="update", ylabel="imagined shaped return",
                        out_path=config.RESULTS_DIR / "dream_train.png")
    except Exception as e:
        print(f"[dream] plot skipped: {e}")
    print(f"DREAM_OK updates={u} "
          f"final_dream_return={log['dream_return'][-1] if log['dream_return'] else 0:+.3f} "
          f"ckpt={out} wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
