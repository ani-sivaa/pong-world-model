"""World model: 4-frame stack + action -> next frame (+ reward & done heads in v2).

Architecture: conv encoder (64->32->16->8->4 spatial), action embedding broadcast
and fused at the 4x4 bottleneck, U-Net-style decoder with encoder skips back to
a 64x64 logit map. Skips let capacity focus on dynamics (ball, paddles) while
static content (walls) rides through nearly free.

Output is LOGITS; training uses BCE (Pong pixels are near-binary), rollout feeds
sigmoid probabilities back in — the same convention used during multi-step
training, so train and dream distributions match.

``load_wm`` remains the checkpoint-compatible single-model loader.  New code
can use ``load_wm_ensemble`` and ``WorldModelEnsemble.predict`` to obtain both
member predictions and aggregate uncertainty without changing old callers.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

import config


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(8, c)


class _Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1), _gn(cout), nn.SiLU())

    def forward(self, x):
        return self.f(x)


class _Up(nn.Module):
    """ConvTranspose upsample, concat skip, 3x3 conv."""

    def __init__(self, cin, cout, skip_c):
        super().__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), _gn(cout), nn.SiLU())
        self.merge = nn.Sequential(nn.Conv2d(cout + skip_c, cout, 3, 1, 1), _gn(cout), nn.SiLU())

    def forward(self, x, skip):
        x = self.up(x)
        return self.merge(torch.cat([x, skip], dim=1))


class WorldModel(nn.Module):
    def __init__(self, with_heads: bool = False, cfg: dict = config.WM):
        super().__init__()
        b = cfg["base_channels"]  # 32
        ae = cfg["act_embed"]
        self.with_heads = with_heads
        self.model_type = "deterministic"
        self.model_config = dict(cfg)

        self.e1 = _Down(config.FRAME_STACK, b)      # 64 -> 32
        self.e2 = _Down(b, b * 2)                   # 32 -> 16
        self.e3 = _Down(b * 2, b * 4)               # 16 -> 8
        self.e4 = _Down(b * 4, b * 8)               # 8  -> 4

        self.act_emb = nn.Embedding(config.N_ACTIONS, ae)
        self.fuse = nn.Sequential(
            nn.Conv2d(b * 8 + ae, b * 8, 3, 1, 1), _gn(b * 8), nn.SiLU(),
            nn.Conv2d(b * 8, b * 8, 3, 1, 1), _gn(b * 8), nn.SiLU(),
        )

        self.d3 = _Up(b * 8, b * 4, skip_c=b * 4)   # 4  -> 8
        self.d2 = _Up(b * 4, b * 2, skip_c=b * 2)   # 8  -> 16
        self.d1 = _Up(b * 2, b, skip_c=b)           # 16 -> 32
        self.d0 = nn.Sequential(nn.ConvTranspose2d(b, b, 4, 2, 1), _gn(b), nn.SiLU())  # 32 -> 64
        self.out = nn.Conv2d(b, 1, 3, 1, 1)         # logits

        if with_heads:
            self.reward_head = nn.Sequential(
                nn.Linear(b * 8, 128), nn.SiLU(), nn.Linear(128, 1))
            self.done_head = nn.Sequential(
                nn.Linear(b * 8, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, stack: torch.Tensor, action: torch.Tensor):
        """stack: float [B,4,64,64] in [0,1] (oldest->newest); action: long [B].

        Returns (frame_logits [B,1,64,64], reward [B] or None, done_logit [B] or None).
        """
        s1 = self.e1(stack)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)

        a = self.act_emb(action)                       # [B, ae]
        a = a[:, :, None, None].expand(-1, -1, s4.shape[2], s4.shape[3])
        z = self.fuse(torch.cat([s4, a], dim=1))       # [B, 256, 4, 4]

        x = self.d3(z, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        logits = self.out(self.d0(x))

        reward = done_logit = None
        if self.with_heads:
            pooled = z.mean(dim=(2, 3))                # [B, 256]
            reward = self.reward_head(pooled).squeeze(-1)
            done_logit = self.done_head(pooled).squeeze(-1)
        return logits, reward, done_logit


class StochasticWorldModel(WorldModel):
    """Conditional-VAE dynamics model with a learned action-conditioned prior.

    The prior sees only the observed frame stack and action. During training the
    posterior additionally sees the true next frame; its latent is injected at
    the dynamics bottleneck, before decoding. At inference ``latent_mode=mean``
    is deterministic and ``sample`` draws a coherent dynamics latent.
    """

    def __init__(self, with_heads: bool = False, cfg: dict = config.WM):
        super().__init__(with_heads=with_heads, cfg=cfg)
        b = cfg["base_channels"]
        latent_dim = int(cfg.get("latent_dim", config.WM.get("latent_dim", 16)))
        self.model_type = "stochastic"
        self.latent_dim = latent_dim
        self.model_config = dict(cfg, latent_dim=latent_dim)

        # Encode the target independently, then condition q on both target and
        # the action-conditioned dynamics context. This is a CVAE rather than
        # output-space noise.
        self.target_encoder = nn.Sequential(
            _Down(1, b), _Down(b, b * 2), _Down(b * 2, b * 4),
            _Down(b * 4, b * 8),
        )
        context_dim = b * 8
        self.prior_stats = nn.Linear(context_dim, 2 * latent_dim)
        self.posterior_stats = nn.Linear(2 * context_dim, 2 * latent_dim)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, context_dim), nn.SiLU(),
        )

    @staticmethod
    def _stats(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = raw.chunk(2, dim=-1)
        return mean, logvar.clamp(-10.0, 10.0)

    @staticmethod
    def _sample(mean: torch.Tensor, logvar: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
        if generator is None:
            noise = torch.randn_like(mean)
        else:
            # Rollout generators are CPU generators so sampling remains
            # reproducible across CPU/CUDA/MPS execution.
            noise = torch.randn(mean.shape, generator=generator, device="cpu",
                                dtype=mean.dtype).to(mean.device)
        return mean + torch.exp(0.5 * logvar) * noise

    def _encode_context(self, stack: torch.Tensor, action: torch.Tensor):
        s1 = self.e1(stack)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        a = self.act_emb(action)
        a = a[:, :, None, None].expand(-1, -1, s4.shape[2], s4.shape[3])
        context = self.fuse(torch.cat([s4, a], dim=1))
        return context, (s1, s2, s3)

    def _decode(self, context: torch.Tensor, latent: torch.Tensor, skips):
        latent_map = self.latent_proj(latent)[:, :, None, None]
        dynamics = context + latent_map
        s1, s2, s3 = skips
        x = self.d3(dynamics, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        logits = self.out(self.d0(x))
        reward = done_logit = None
        if self.with_heads:
            pooled = dynamics.mean(dim=(2, 3))
            reward = self.reward_head(pooled).squeeze(-1)
            done_logit = self.done_head(pooled).squeeze(-1)
        return logits, reward, done_logit

    def forward(self, stack: torch.Tensor, action: torch.Tensor,
                latent_mode: str = "mean",
                generator: torch.Generator | None = None):
        """Infer from p(z|stack, action), deterministically or by sampling."""
        context, skips = self._encode_context(stack, action)
        prior_mean, prior_logvar = self._stats(
            self.prior_stats(context.mean(dim=(2, 3))))
        if latent_mode == "mean":
            latent = prior_mean
        elif latent_mode == "sample":
            latent = self._sample(prior_mean, prior_logvar, generator)
        else:
            raise ValueError("latent_mode must be 'mean' or 'sample'")
        return self._decode(context, latent, skips)

    def forward_train(self, stack: torch.Tensor, action: torch.Tensor,
                      target: torch.Tensor, sample_posterior: bool = True):
        """Decode q(z|stack, action, target) and return per-example KL terms."""
        if target.ndim == 3:
            target = target[:, None]
        context, skips = self._encode_context(stack, action)
        pooled = context.mean(dim=(2, 3))
        prior_mean, prior_logvar = self._stats(self.prior_stats(pooled))
        target_features = self.target_encoder(target).mean(dim=(2, 3))
        post_mean, post_logvar = self._stats(
            self.posterior_stats(torch.cat([pooled, target_features], dim=-1)))
        latent = (self._sample(post_mean, post_logvar)
                  if sample_posterior else post_mean)
        output = self._decode(context, latent, skips)
        # KL[q(z|x,y) || p(z|x)], retained per dimension for free-bits.
        kl_per_dim = 0.5 * (
            prior_logvar - post_logvar
            + (post_logvar.exp() + (post_mean - prior_mean).square())
            / prior_logvar.exp()
            - 1.0
        )
        return (*output, kl_per_dim)


def load_wm(ckpt_path, device, with_heads=False) -> WorldModel:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    saved_cfg = ckpt.get("config", {})
    model_type = ckpt.get("model_type", saved_cfg.get("model_type"))
    if model_type is None:
        model_type = ("stochastic" if any(
            key.startswith(("prior_stats.", "posterior_stats.", "target_encoder."))
            for key in ckpt["model"]) else "deterministic")
    if model_type not in ("deterministic", "stochastic"):
        raise ValueError(f"unsupported world-model type: {model_type!r}")
    model_cfg = dict(config.WM)
    for key in ("base_channels", "act_embed", "latent_dim"):
        if key in saved_cfg:
            model_cfg[key] = saved_cfg[key]
    model_cls = StochasticWorldModel if model_type == "stochastic" else WorldModel
    wm = model_cls(with_heads=with_heads, cfg=model_cfg)
    missing, unexpected = wm.load_state_dict(ckpt["model"], strict=False)
    # strict=False covers both directions: v1 ckpt seeding a v2 model (head keys
    # missing -> fresh init) and v2 ckpt loaded headless for drift eval (head
    # keys unexpected -> ignored). Anything else is a real corruption.
    _heads = ("reward_head", "done_head")
    assert all(k.startswith(_heads) for k in unexpected), f"unexpected: {unexpected}"
    assert all(k.startswith(_heads) for k in missing), f"missing: {missing}"
    return wm.to(device)


@dataclass
class EnsemblePrediction:
    """One-step member predictions and aggregate disagreement.

    Frames and done values are probabilities.  Scalar disagreement metrics are
    shaped ``[B]``; ``frame_variance`` is mean per-pixel variance and
    ``frame_mse`` is mean pairwise frame MSE (zero for a single member).
    """

    member_frames: torch.Tensor       # [M,B,1,H,W]
    member_rewards: torch.Tensor      # [M,B]
    member_done_probs: torch.Tensor   # [M,B]
    mean_frame: torch.Tensor          # [B,1,H,W]
    mean_reward: torch.Tensor         # [B]
    mean_done_prob: torch.Tensor      # [B]
    frame_variance: torch.Tensor      # [B]
    frame_mse: torch.Tensor           # [B]
    reward_std: torch.Tensor          # [B]
    reward_variance: torch.Tensor     # [B]
    done_disagreement: torch.Tensor   # [B]
    aleatoric_frame_mse: torch.Tensor # [B], sampled-prior vs prior-mean
    aleatoric_reward_mse: torch.Tensor # [B]
    aleatoric_done_mse: torch.Tensor   # [B]
    rollout_reward: torch.Tensor       # [B], coherent with rollout frame
    rollout_done_prob: torch.Tensor    # [B], coherent with rollout frame


class WorldModelEnsemble(nn.Module):
    """Frozen-or-trainable collection of checkpoint-compatible world models."""

    def __init__(self, members: Sequence[WorldModel]):
        super().__init__()
        if not members:
            raise ValueError("world-model ensemble requires at least one member")
        self.members = nn.ModuleList(members)

    def forward(self, stack: torch.Tensor, action: torch.Tensor) -> EnsemblePrediction:
        """Alias for :meth:`predict`, convenient for evaluators."""
        return self.predict(stack, action)

    def predict(self, stack: torch.Tensor, action: torch.Tensor) -> EnsemblePrediction:
        # Epistemic metrics must use one deterministic prediction per member.
        raw = [member(stack, action, latent_mode="mean")
               if isinstance(member, StochasticWorldModel)
               else member(stack, action)
               for member in self.members]
        if any(reward is None or done is None for _, reward, done in raw):
            raise RuntimeError("ensemble predictions require checkpoints loaded with heads")
        frames = torch.stack([torch.sigmoid(logits) for logits, _, _ in raw])
        rewards = torch.stack([reward for _, reward, _ in raw])
        done_probs = torch.stack([torch.sigmoid(done) for _, _, done in raw])

        mean_frame = frames.mean(0)
        mean_reward = rewards.mean(0)
        mean_done = done_probs.mean(0)
        frame_var_pixels = frames.var(0, unbiased=False)
        frame_variance = frame_var_pixels.mean(dim=(1, 2, 3))
        reward_variance = rewards.var(0, unbiased=False)
        done_disagreement = done_probs.var(0, unbiased=False)
        if len(self.members) == 1:
            frame_mse = torch.zeros_like(frame_variance)
        else:
            # Mean MSE over all unordered member pairs.
            pair_mse = [
                (frames[i] - frames[j]).square().mean(dim=(1, 2, 3))
                for i in range(len(self.members))
                for j in range(i + 1, len(self.members))
            ]
            frame_mse = torch.stack(pair_mse).mean(0)
        return EnsemblePrediction(
            member_frames=frames,
            member_rewards=rewards,
            member_done_probs=done_probs,
            mean_frame=mean_frame,
            mean_reward=mean_reward,
            mean_done_prob=mean_done,
            frame_variance=frame_variance,
            frame_mse=frame_mse,
            reward_std=reward_variance.sqrt(),
            reward_variance=reward_variance,
            done_disagreement=done_disagreement,
            aleatoric_frame_mse=torch.zeros_like(frame_variance),
            aleatoric_reward_mse=torch.zeros_like(frame_variance),
            aleatoric_done_mse=torch.zeros_like(frame_variance),
            rollout_reward=mean_reward,
            rollout_done_prob=mean_done,
        )

    def rollout_step(self, stack: torch.Tensor, action: torch.Tensor,
                     frame_mode: str = "mean",
                     latent_mode: str = "mean",
                     generator: torch.Generator | None = None):
        """Predict one step and return ``(next_stack, prediction)``.

        ``mean`` is deterministic. ``sample`` draws one member independently
        per batch item using ``generator``; this samples coherent member frames,
        rather than noisy pixels, and is reproducible with a seeded generator.
        """
        prediction = self.predict(stack, action)
        rollout_frames = prediction.member_frames
        rollout_rewards = prediction.member_rewards
        rollout_done_probs = prediction.member_done_probs
        if latent_mode == "sample":
            sampled = [
                member(stack, action, latent_mode="sample", generator=generator)
                if isinstance(member, StochasticWorldModel)
                else (None, None, None)
                for index, member in enumerate(self.members)
            ]
            rollout_frames = torch.stack([
                torch.sigmoid(output[0]) if output[0] is not None
                else prediction.member_frames[index]
                for index, output in enumerate(sampled)
            ])
            rollout_rewards = torch.stack([
                output[1] if output[1] is not None
                else prediction.member_rewards[index]
                for index, output in enumerate(sampled)
            ])
            rollout_done_probs = torch.stack([
                torch.sigmoid(output[2]) if output[2] is not None
                else prediction.member_done_probs[index]
                for index, output in enumerate(sampled)
            ])
            aleatoric = (rollout_frames - prediction.member_frames).square().mean(
                dim=(0, 2, 3, 4))
            prediction = replace(
                prediction,
                aleatoric_frame_mse=aleatoric,
                aleatoric_reward_mse=(
                    rollout_rewards - prediction.member_rewards).square().mean(0),
                aleatoric_done_mse=(
                    rollout_done_probs
                    - prediction.member_done_probs).square().mean(0),
            )
        elif latent_mode != "mean":
            raise ValueError("latent_mode must be 'mean' or 'sample'")
        if frame_mode == "mean":
            next_frame = rollout_frames.mean(0)
            rollout_reward = rollout_rewards.mean(0)
            rollout_done_prob = rollout_done_probs.mean(0)
        elif frame_mode == "sample":
            batch = stack.shape[0]
            member_index = torch.randint(
                len(self.members), (batch,), device="cpu", generator=generator
            ).to(stack.device)
            batch_index = torch.arange(batch, device=stack.device)
            next_frame = rollout_frames[member_index, batch_index]
            rollout_reward = rollout_rewards[member_index, batch_index]
            rollout_done_prob = rollout_done_probs[member_index, batch_index]
        else:
            raise ValueError("frame_mode must be 'mean' or 'sample'")
        prediction = replace(
            prediction, rollout_reward=rollout_reward,
            rollout_done_prob=rollout_done_prob)
        return torch.cat([stack[:, 1:], next_frame], dim=1), prediction


def load_wm_ensemble(ckpt_paths, device, with_heads=True) -> WorldModelEnsemble:
    """Load one or more ordinary WM checkpoints as an ensemble.

    A string/Path is treated as a one-member ensemble.  Checkpoints are loaded
    solely through ``load_wm``, preserving old v1/v2 compatibility behavior.
    """
    if isinstance(ckpt_paths, (str, Path)):
        ckpt_paths = [ckpt_paths]
    paths = list(ckpt_paths)
    if not paths:
        raise ValueError("at least one world-model checkpoint is required")
    return WorldModelEnsemble(
        [load_wm(path, device, with_heads=with_heads) for path in paths])
