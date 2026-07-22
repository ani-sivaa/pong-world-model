"""World model: 4-frame stack + action -> next frame (+ reward & done heads in v2).

Architecture: conv encoder (64->32->16->8->4 spatial), action embedding broadcast
and fused at the 4x4 bottleneck, U-Net-style decoder with encoder skips back to
a 64x64 logit map. Skips let capacity focus on dynamics (ball, paddles) while
static content (walls) rides through nearly free.

Output is LOGITS; training uses BCE (Pong pixels are near-binary), rollout feeds
sigmoid probabilities back in — the same convention used during multi-step
training, so train and dream distributions match.
"""
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


def load_wm(ckpt_path, device, with_heads=False) -> WorldModel:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    wm = WorldModel(with_heads=with_heads)
    missing, unexpected = wm.load_state_dict(ckpt["model"], strict=False)
    # strict=False covers both directions: v1 ckpt seeding a v2 model (head keys
    # missing -> fresh init) and v2 ckpt loaded headless for drift eval (head
    # keys unexpected -> ignored). Anything else is a real corruption.
    _heads = ("reward_head", "done_head")
    assert all(k.startswith(_heads) for k in unexpected), f"unexpected: {unexpected}"
    assert all(k.startswith(_heads) for k in missing), f"missing: {missing}"
    return wm.to(device)
