"""Shared vision policy — IDENTICAL architecture for the PPO baseline (3a) and
the dream-trained agent (3b), so the transfer comparison isolates the training
regime, not the network. Nature-CNN scaled to 64x64 grayscale stacks.
"""
import torch
import torch.nn as nn

import config


def _ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class PolicyNet(nn.Module):
    def __init__(self, cfg: dict = config.POLICY):
        super().__init__()
        c1, c2, c3 = cfg["channels"]
        fc = cfg["fc"]
        self.trunk = nn.Sequential(
            _ortho(nn.Conv2d(config.FRAME_STACK, c1, 8, 4), 2 ** 0.5), nn.ReLU(),  # 64->15
            _ortho(nn.Conv2d(c1, c2, 4, 2), 2 ** 0.5), nn.ReLU(),                  # 15->6
            _ortho(nn.Conv2d(c2, c3, 3, 1), 2 ** 0.5), nn.ReLU(),                  # 6->4
            nn.Flatten(),
            _ortho(nn.Linear(c3 * 4 * 4, fc), 2 ** 0.5), nn.ReLU(),
        )
        self.pi = _ortho(nn.Linear(fc, config.N_ACTIONS), 0.01)
        self.v = _ortho(nn.Linear(fc, 1), 1.0)

    def forward(self, x):
        """x: float [B,4,64,64] in [0,1] -> (logits [B,3], value [B])."""
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


class LogitsOnly(nn.Module):
    """ONNX-export wrapper: frames -> logits (the web demo only needs argmax)."""

    def __init__(self, policy: PolicyNet):
        super().__init__()
        self.policy = policy

    def forward(self, x):
        return self.policy(x)[0]


def load_policy(ckpt_path, device) -> PolicyNet:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    p = PolicyNet()
    p.load_state_dict(ckpt["model"])
    return p.to(device)
