"""World-model trainer.

Multi-step unroll: after a 1-step warmup phase, the model is unrolled k steps
feeding its own sigmoid outputs back in WITH gradient flowing through the
unroll — this is what makes long autoregressive dreams stable.

Change-weighted BCE: per-pixel weight = 1 + w * |frame_{t+j} - frame_{t+j-1}|
(computed on GROUND-TRUTH frames) — without it the tiny 2x2 ball contributes
~0.1% of the loss and blurs out of existence in rollouts.

CLI:
  python -m wm.train --scale full [--v2] [--init-from ckpt] [--data-frac 0.25]
      [--tag ablation25] [--seed 42] [--bootstrap] [--bootstrap-frac 1.0]
      [--steps N] [--max-seconds S] [--out path.pt]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from wm.data import EVENT_NAMES, TransitionData
from wm.model import StochasticWorldModel, WorldModel, load_wm


def to_dev(x, dev, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(x)).to(dev, dtype)


def interior_ball_mass(frames: torch.Tensor) -> torch.Tensor:
    """Sum of soft ball pixels, excluding walls and paddle columns.

    Matches ``scripts.evaluate_trust._has_ball``'s interior mask so the
    ballless-reward training penalty targets the same invariant.
    """
    if frames.ndim == 4:
        frames = frames[:, 0]
    mass_map = frames.clone()
    mass_map[:, 0, :] = 0
    mass_map[:, config.ENV["H"] - 1, :] = 0
    for x in (config.ENV["left_x"], config.ENV["right_x"]):
        mass_map[:, :, x:x + config.ENV["paddle_w"]] = 0
    return mass_map.sum(dim=(1, 2))


def event_balanced_weights(events: torch.Tensor, alive: torch.Tensor,
                           clip: float = 20.0) -> torch.Tensor:
    """Inverse-frequency event weights, normalized over valid examples."""
    if events.shape != alive.shape:
        raise ValueError("events and alive must have identical [B,k] shapes")
    result = torch.zeros_like(alive)
    for step in range(events.shape[1]):
        active = alive[:, step] > 0
        if not active.any():
            continue
        labels = events[active, step].long()
        counts = torch.bincount(labels)
        weights = counts.sum().float() / counts.clamp_min(1).float()
        selected = weights[labels].clamp_max(clip)
        selected = selected / selected.mean().clamp_min(1e-8)
        result[active, step] = selected
    return result


def compute_losses(wm, batch, dev, v2, w_change, kl_coef=None,
                   free_bits=None, return_kl=False, events=None,
                   utilization_coef=None, utilization_target=None,
                   ballless_reward_coef=None, ballless_mass_tau=None):
    stack_u8, acts, targets_u8, rews, dns, alive = batch
    B, k = acts.shape
    stack = to_dev(stack_u8, dev) / 255.0
    targets = to_dev(targets_u8, dev) / 255.0            # [B,k,64,64]
    acts_t = to_dev(acts, dev, torch.long)
    alive_t = to_dev(alive, dev)                          # [B,k] episode alive at step j
    dns_t = to_dev(dns, dev)                              # [B,k] done fires AT step j
    head_weights = alive_t
    if v2 and events is not None and config.WM.get("head_event_balance", False):
        events_t = to_dev(events, dev, torch.long)
        head_weights = event_balanced_weights(
            events_t, alive_t, float(config.WM.get("head_weight_clip", 20.0)))

    # ground-truth previous frames for change weights: [stack[-1], targets[:-1]]
    prev = torch.cat([stack[:, -1:], targets[:, :-1]], dim=1)

    frame_loss = torch.zeros((), device=dev)
    rew_loss = torch.zeros((), device=dev)
    done_loss = torch.zeros((), device=dev)
    kl_loss = torch.zeros((), device=dev)
    util_loss = torch.zeros((), device=dev)
    ballless_loss = torch.zeros((), device=dev)
    is_stochastic = isinstance(wm, StochasticWorldModel)
    kl_coef = config.WM["kl_coef"] if kl_coef is None else kl_coef
    free_bits = config.WM["free_bits"] if free_bits is None else free_bits
    utilization_coef = (
        config.WM.get("utilization_coef", 0.0) if utilization_coef is None
        else utilization_coef)
    utilization_target = (
        config.WM.get("utilization_target", 0.0) if utilization_target is None
        else utilization_target)
    ballless_reward_coef = (
        config.WM.get("ballless_reward_coef", 0.0) if ballless_reward_coef is None
        else ballless_reward_coef)
    ballless_mass_tau = (
        config.WM.get("ballless_mass_tau", 0.75) if ballless_mass_tau is None
        else ballless_mass_tau)
    cur = stack
    for j in range(k):
        if is_stochastic:
            logits, r, d, kl_per_dim = wm.forward_train(
                cur, acts_t[:, j], targets[:, j],
                sample_posterior=wm.training)
        else:
            logits, r, d = wm(cur, acts_t[:, j])
        weight = 1.0 + w_change * (targets[:, j] - prev[:, j]).abs()
        bce = F.binary_cross_entropy_with_logits(
            logits[:, 0], targets[:, j], weight=weight, reduction="none")
        # frame target t+j+1 is valid only if the episode is alive at step j
        # AND does not end AT step j (frames[t+j+1] would be a reset frame,
        # which depends on serve RNG — unlearnable and semantically wrong)
        frame_mask = alive_t[:, j] * (1.0 - dns_t[:, j])
        frame_loss = frame_loss + (bce.mean(dim=(1, 2)) * frame_mask).mean()
        if is_stochastic:
            # Free bits prevent weak posterior dimensions from being rewarded
            # for collapsing completely to the prior.
            per_example_kl = kl_per_dim.clamp_min(free_bits).sum(dim=-1)
            kl_loss = kl_loss + (per_example_kl * frame_mask).mean()
            if utilization_coef > 0 and wm.training:
                # Hinge: keep prior samples disagreeing in pixel space. Trust
                # measures sample-to-sample variance; late training previously
                # drove that below 1e-7 while prior_std looked healthy.
                sample_a = torch.sigmoid(wm(
                    cur, acts_t[:, j], latent_mode="sample")[0])
                sample_b = torch.sigmoid(wm(
                    cur, acts_t[:, j], latent_mode="sample")[0])
                sample_var = (sample_a - sample_b).square().mean(dim=(1, 2, 3))
                util_loss = util_loss + (
                    F.relu(utilization_target - sample_var) * frame_mask
                ).mean()
        if v2:
            # reward/done AT step j are valid whenever the episode is alive at j
            rt = to_dev(rews[:, j], dev)
            rew_loss = rew_loss + (F.mse_loss(r, rt, reduction="none")
                                   * head_weights[:, j]).mean()
            done_loss = done_loss + (F.binary_cross_entropy_with_logits(
                d, dns_t[:, j], reduction="none") * head_weights[:, j]).mean()
            if ballless_reward_coef > 0:
                pred_mass = interior_ball_mass(torch.sigmoid(logits[:, 0]))
                ballless_w = torch.exp(-pred_mass / max(ballless_mass_tau, 1e-6))
                ballless_loss = ballless_loss + (
                    F.relu(r) * ballless_w * alive_t[:, j]
                ).mean()
        if j + 1 < k:  # feed own prediction back in, gradient intact
            cur = torch.cat([cur[:, 1:], torch.sigmoid(logits)], dim=1)

    frame_loss = frame_loss / k
    kl_loss = kl_loss / k
    util_loss = util_loss / k
    ballless_loss = ballless_loss / k
    total = frame_loss + (kl_coef * kl_loss if is_stochastic else 0.0)
    if is_stochastic and utilization_coef > 0:
        total = total + utilization_coef * util_loss
    if v2:
        rew_loss, done_loss = rew_loss / k, done_loss / k
        total = total + config.WM["reward_loss_weight"] * rew_loss \
                      + config.WM["done_loss_weight"] * done_loss
        if ballless_reward_coef > 0:
            total = total + ballless_reward_coef * ballless_loss
    losses = (total, frame_loss, rew_loss, done_loss)
    if return_kl:
        return (*losses, kl_loss, util_loss, ballless_loss)
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="local", choices=list(config.SCALES))
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--stochastic", action="store_true",
                    help="train a conditional latent (CVAE) world model")
    ap.add_argument("--latent-dim", type=int, default=config.WM["latent_dim"])
    ap.add_argument("--kl-coef", type=float, default=config.WM["kl_coef"])
    ap.add_argument("--kl-warmup-frac", type=float,
                    default=config.WM["kl_warmup_frac"])
    ap.add_argument("--free-bits", type=float, default=config.WM["free_bits"])
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--data-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=config.SEED,
                    help="model initialization and sampler RNG seed")
    ap.add_argument("--bootstrap", action=argparse.BooleanOptionalAction,
                    default=config.FLYWHEEL["bootstrap"],
                    help="fixed seed-specific bootstrap over valid train windows")
    ap.add_argument("--bootstrap-frac", type=float,
                    default=config.FLYWHEEL["bootstrap_frac"],
                    help="bootstrap draws as a fraction of valid windows")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-seconds", type=int, default=config.CAPS["wm_train"])
    ap.add_argument("--sampler", default=config.FLYWHEEL["sampler"],
                    choices=("natural", "balanced", "priority"),
                    help="training-window sampler; validation is always natural")
    args = ap.parse_args()

    config.seed_everything(args.seed)
    dev = config.get_device()
    sc = config.SCALES[args.scale]
    steps = args.steps or sc["wm_steps"]
    batch = sc["wm_batch"]
    K = config.WM["unroll_k"]
    warmup_end = max(1, int(steps * config.WM["warmup_frac"]))
    tag = args.tag or (
        "wm_stochastic_v2" if args.stochastic and args.v2
        else "wm_stochastic" if args.stochastic
        else "wm_v2" if args.v2 else "wm_v1")
    out = Path(args.out or (config.CKPT_DIR / f"{tag}.pt"))
    out.parent.mkdir(parents=True, exist_ok=True)
    data_dir = args.data or (config.DATA_DIR / args.scale)

    data = TransitionData(
        data_dir, frac=args.data_frac, seed=args.seed,
        bootstrap=args.bootstrap, bootstrap_frac=args.bootstrap_frac)
    print(f"[wm.train] tag={tag} dev={dev.type} steps={steps} batch={batch} "
          f"T={data.T} train_end={data.train_end} v2={args.v2} "
          f"frac={args.data_frac} sampler={args.sampler} "
          f"seed={args.seed} bootstrap={args.bootstrap} "
          f"bootstrap_frac={args.bootstrap_frac} "
          f"model_type={'stochastic' if args.stochastic else 'deterministic'} "
          f"events={data.has_events} priorities={data.has_priorities}", flush=True)

    model_cfg = dict(config.WM, latent_dim=args.latent_dim,
                     kl_coef=args.kl_coef, free_bits=args.free_bits,
                     kl_warmup_frac=args.kl_warmup_frac)
    if args.init_from:
        wm = load_wm(args.init_from, dev, with_heads=args.v2)
        if args.stochastic != isinstance(wm, StochasticWorldModel):
            raise ValueError("--stochastic must match --init-from checkpoint model type")
        model_cfg = dict(wm.model_config, kl_coef=args.kl_coef,
                         free_bits=args.free_bits)
    else:
        model_cls = StochasticWorldModel if args.stochastic else WorldModel
        wm = model_cls(with_heads=args.v2, cfg=model_cfg).to(dev)
    opt = torch.optim.AdamW(wm.parameters(), lr=config.WM["lr"],
                            weight_decay=config.WM["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    log = {"step": [], "loss": [], "frame": [], "reward": [], "done": [], "kl": [],
           "kl_beta": [], "prior_std": [], "posterior_prior_kl": [],
           "latent_utilization": [], "util_loss": [], "ballless_loss": [],
           "batch_events": [], "val_frame": [], "val_step": [],
           "val_event_frame": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        k = 1 if step <= warmup_end else K
        b, batch_info = data.sample(
            batch, k, sampler=args.sampler, return_info=True)
        kl_warmup_steps = max(1, int(steps * args.kl_warmup_frac))
        kl_beta = args.kl_coef * min(1.0, step / kl_warmup_steps)
        with torch.amp.autocast(dev.type, enabled=use_amp):
            total, fl, rl, dl, kl, util_l, ballless_l = compute_losses(
                wm, b, dev, args.v2, config.WM["change_loss_weight"],
                kl_coef=kl_beta, free_bits=args.free_bits, return_kl=True,
                events=batch_info["events"])
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(wm.parameters(), config.WM["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % 100 == 0 or step == 1:
            event_count = np.bincount(
                batch_info["primary_events"], minlength=max(EVENT_NAMES) + 1)
            event_log = {
                EVENT_NAMES[i]: int(event_count[i])
                for i in range(len(event_count)) if event_count[i]}
            log["step"].append(step)
            log["loss"].append(float(total.detach()))
            log["frame"].append(float(fl.detach()))
            log["reward"].append(float(rl.detach()))
            log["done"].append(float(dl.detach()))
            log["kl"].append(float(kl.detach()))
            log["kl_beta"].append(float(kl_beta))
            log["util_loss"].append(float(util_l.detach()))
            log["ballless_loss"].append(float(ballless_l.detach()))
            if isinstance(wm, StochasticWorldModel):
                with torch.no_grad():
                    diagnostic_n = min(16, len(b[0]))
                    diagnostic_stack = to_dev(b[0][:diagnostic_n], dev) / 255.0
                    diagnostic_action = to_dev(
                        b[1][:diagnostic_n, 0], dev, torch.long)
                    diagnostic_target = to_dev(
                        b[2][:diagnostic_n, 0], dev) / 255.0
                    stats = wm.latent_statistics(
                        diagnostic_stack, diagnostic_action, diagnostic_target)
                    mean_frame = torch.sigmoid(wm(
                        diagnostic_stack, diagnostic_action,
                        latent_mode="mean")[0])
                    sample_frame = torch.sigmoid(wm(
                        diagnostic_stack, diagnostic_action,
                        latent_mode="sample")[0])
                    log["prior_std"].append(float(
                        (0.5 * stats["prior_logvar"]).exp().mean()))
                    log["posterior_prior_kl"].append(float(
                        stats["posterior_prior_kl"].sum(-1).mean()))
                    log["latent_utilization"].append(float(
                        (sample_frame - mean_frame).square().mean()))
            else:
                for name in ("prior_std", "posterior_prior_kl",
                             "latent_utilization"):
                    log[name].append(None)
            log["batch_events"].append(event_log)
            print(f"  step {step}/{steps} k={k} loss={float(total):.5f} "
                  f"frame={float(fl):.5f} rew={float(rl):.5f} "
                  f"done={float(dl):.5f} kl={float(kl):.5f} "
                  f"util={float(util_l):.5f} ballless={float(ballless_l):.5f} "
                  f"events={event_log} {time.time()-t0:.0f}s", flush=True)
        if step % 1000 == 0 or step == steps:
            wm.eval()
            with torch.no_grad(), torch.amp.autocast(dev.type, enabled=use_amp):
                vb, val_info = data.sample(
                    min(batch, 64), K, val=True, sampler="natural",
                    return_info=True)
                _, vfl, _, _, _, _, _ = compute_losses(
                    wm, vb, dev, args.v2, config.WM["change_loss_weight"],
                    kl_coef=args.kl_coef, free_bits=args.free_bits, return_kl=True)
                per_event = {}
                for code in np.unique(val_info["primary_events"]):
                    mask = val_info["primary_events"] == code
                    if not mask.any():
                        continue
                    event_batch = tuple(x[mask] for x in vb)
                    _, event_fl, _, _, _, _, _ = compute_losses(
                        wm, event_batch, dev, args.v2,
                        config.WM["change_loss_weight"], kl_coef=args.kl_coef,
                        free_bits=args.free_bits, return_kl=True)
                    per_event[EVENT_NAMES.get(int(code), str(int(code)))] = \
                        float(event_fl)
            wm.train()
            log["val_frame"].append(float(vfl))
            log["val_step"].append(step)
            log["val_event_frame"].append(per_event)
            torch.save({"model": {k_: v.cpu() for k_, v in wm.state_dict().items()},
                        "model_type": wm.model_type,
                        "config": dict(model_cfg, v2=args.v2,
                                       model_type=wm.model_type,
                                       data_frac=args.data_frac,
                                       sampler=args.sampler, seed=args.seed,
                                       bootstrap=args.bootstrap,
                                       bootstrap_frac=args.bootstrap_frac),
                        "step": step}, out)
            print(f"  [ckpt] step {step} val_frame={float(vfl):.5f} "
                  f"val_events={per_event} -> {out}",
                  flush=True)
        if time.time() - t0 > args.max_seconds:
            print(f"[wm.train] WALL-CLOCK CAP hit at step {step} "
                  f"({args.max_seconds}s) — saving and exiting", flush=True)
            break

    torch.save({"model": {k_: v.cpu() for k_, v in wm.state_dict().items()},
                "model_type": wm.model_type,
                "config": dict(model_cfg, v2=args.v2,
                               model_type=wm.model_type,
                               data_frac=args.data_frac, sampler=args.sampler,
                               seed=args.seed, bootstrap=args.bootstrap,
                               bootstrap_frac=args.bootstrap_frac),
                "step": step}, out)
    log_path = config.RESULTS_DIR / f"wm_train_log_{tag}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log))
    try:
        from evalutils import plots
        plots.line_plot(log["step"], {"train frame loss": log["frame"]},
                        title=f"World-model training ({tag})", xlabel="step",
                        ylabel="weighted BCE",
                        out_path=config.RESULTS_DIR / f"wm_train_{tag}.png",
                        log_y=True)
    except Exception as e:  # plotting must never kill a finished train
        print(f"[wm.train] plot skipped: {e}")
    print(f"WM_TRAIN_OK tag={tag} steps={step} "
          f"final_val={log['val_frame'][-1] if log['val_frame'] else float(fl):.5f} "
          f"wall={time.time()-t0:.0f}s ckpt={out}")


if __name__ == "__main__":
    main()
