"""Single source of truth for EVERY hyperparameter and constant in the project.

Import cheaply: this module must import with stdlib only (no torch/numpy) so the
env, the constants exporter, and web tooling can use it anywhere.

Scale presets: "smoke" (preflight), "local" (CPU/MPS fallback), "full" (Modal GPU).
Select with --scale on every entrypoint; code reads SCALES[scale].
"""
import os
from pathlib import Path

SEED = 42

# ---------------------------------------------------------------- paths ----
# On Modal, set WM_ROOT=/vol so all artifacts land on the persistent volume.
REPO_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("WM_ROOT", str(REPO_ROOT)))
DATA_DIR = ROOT / "data"           # data/<scale>/{frames,actions,rewards,dones}.npy + meta.json
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

# ------------------------------------------------------------------ env ----
# Pixel-exact contract shared by pong/env.py (Python) and web/pong.js (JS).
# Rounding convention everywhere: floor(x + 0.5)  (Python round() banker-rounds;
# JS Math.round differs at .5 — floor(x+0.5) is identical in both languages).
ENV = dict(
    W=64, H=64,
    # walls: rows 0 and H-1 are drawn white; ball bounces to stay inside them
    paddle_h=12, paddle_w=2,
    left_x=2,      # left paddle occupies columns [2, 4)
    right_x=60,    # right paddle occupies columns [60, 62)
    agent_speed=2.0,     # right paddle px/step (the learning agent; human in web)
    opp_speed=1.0,       # left scripted paddle px/step (slower => beatable)
    opp_deadzone=1.0,    # opponent only moves if |paddle_cy - ball_cy| > deadzone
    ball_size=2,
    ball_vx=1.5,         # constant |vx|
    max_vy=1.25,         # paddle-hit sets vy = offset_frac * max_vy (> opp_speed => angled shots win)
    serve_vy_choices=(-1.0, -0.5, 0.0, 0.5, 1.0),  # seeded pick at reset; only stochasticity in the env
    max_steps=500,       # truncation => done=True, reward 0
    # rewards (from the RIGHT/agent perspective)
    r_score=1.0,         # opponent missed
    r_concede=-1.0,      # agent missed
    r_hit=0.1,           # shaping: agent paddle contact (evals report raw points separately)
)
N_ACTIONS = 3            # 0=stay, 1=up (y decreases), 2=down (y increases)
FRAME_STACK = 4          # one frame doesn't encode velocity

# Data-collection behavior policy: scripted tracker with eps-random actions
COLLECT = dict(eps_random=0.20, n_envs=64)

# ---------------------------------------------------------- world model ----
WM = dict(
    base_channels=32,        # enc: 32-64-128-256 at 32,16,8,4 spatial
    act_embed=32,
    change_loss_weight=15.0, # per-pixel weight = 1 + w*|next - last|; keeps the tiny ball sharp
    lr=1e-3, weight_decay=1e-5, batch_size=128,
    unroll_k=5,              # multi-step training unroll (feed own sigmoid outputs back)
    warmup_frac=0.35,        # fraction of steps trained at k=1 before k=unroll_k
    grad_clip=1.0,
    val_frac=0.05,           # episode-level split
    # v2 heads (phase 3b)
    reward_loss_weight=1.0, done_loss_weight=1.0,
)

# ------------------------------------------------------------- policies ----
POLICY = dict(               # identical arch for 3a (PPO) and 3b (dream)
    channels=(32, 64, 64), fc=256,
)
PPO = dict(
    n_envs=64, rollout=128, epochs=4, minibatches=8,
    gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.01, vf_coef=0.5,
    lr=2.5e-4, grad_clip=0.5,
)
DREAM = dict(
    horizon=40, batch=256, gamma=0.97,
    ent_coef=0.01, vf_coef=0.5, lr=3e-4, grad_clip=0.5,
)

# -------------------------------------------------------- scale presets ----
SCALES = dict(
    smoke=dict(
        transitions=5_000, wm_steps=200, wm_batch=32,
        ppo_steps=8_192, dream_updates=20,
        rollout_eval_h=20, eval_episodes=5,
    ),
    local=dict(
        transitions=100_000, wm_steps=3_000, wm_batch=64,
        ppo_steps=500_000, dream_updates=600,
        rollout_eval_h=60, eval_episodes=50,
    ),
    full=dict(
        transitions=1_000_000, wm_steps=18_000, wm_batch=128,
        ppo_steps=3_000_000, dream_updates=2_500,
        rollout_eval_h=60, eval_episodes=200,
    ),
)

# ---------------------------------------------------- wall-clock caps (s) --
CAPS = dict(
    collect=1_800, wm_train=7_200, wm_eval=1_800, ablation=5_400,
    ppo=5_400, wm_v2=3_600, dream=5_400, transfer=1_800,
)

ABLATION_FRACS = (0.10, 0.25, 0.50, 1.00)

# ONNX contract (see INTERFACES.md): input "frames" [1,4,64,64] float32 in [0,1],
# output "logits" [1,3]; opset 17.
ONNX = dict(opset=17, input_name="frames", output_name="logits")


def get_device():
    """cuda > mps > cpu. Lazy torch import so config stays stdlib-importable."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = SEED):
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
