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

# Adaptive real-data flywheel. Keep this section stdlib-only: collectors and
# remote launchers import config before numpy/torch are necessarily available.
FLYWHEEL = dict(
    # "natural" is an unmodified tracker stream written last, so the existing
    # episode-level validation split remains natural rather than enriched.
    composition=dict(
        tracker=0.20, baseline=0.20, dream=0.15, stochastic=0.0, redteam=0.10,
        random=0.10, rare=0.15, natural=0.10,
    ),
    natural_holdout_frac=0.10,
    # Multipliers used to seed priorities before optional WM disagreement.
    event_priority=dict(
        ordinary=1.0, hit=5.0, score=10.0, concede=10.0,
        done_truncation=4.0, serve_near_terminal=2.0, boundary=1.0,
        miss=8.0,
    ),
    disagreement_weight=4.0,
    frame_mismatch_weight=1.0,
    reward_mismatch_weight=1.0,
    done_mismatch_weight=1.0,
    priority_floor=1e-3,
    # Pre-terminal frames are the useful historical context for rare outcomes.
    # Their priorities are decayed backwards from each real event anchor.
    terminal_context_radius=8,
    terminal_context_decay=0.75,
    # Relative event mass for inverse-frequency balanced window sampling.
    # Boundary is deliberately zero: forced stream breaks are not game events.
    balanced_event_weight=dict(
        ordinary=1.0, hit=1.0, score=1.0, concede=1.0,
        done_truncation=1.0, serve_near_terminal=1.0, boundary=0.0,
        miss=1.0,
    ),
    sampler="natural",
    ensemble_size=3,
    ensemble_seeds=(42, 314, 2718),
    bootstrap=True,
    bootstrap_frac=1.0,
)

# ---------------------------------------------------------- world model ----
WM = dict(
    base_channels=32,        # enc: 32-64-128-256 at 32,16,8,4 spatial
    act_embed=32,
    latent_dim=16,           # stochastic CVAE dynamics latent
    stochastic_heads_deterministic=True,
    # Prior full-scale rounds collapsed late: prior_std stayed healthy while
    # sample-to-sample frame variance fell below the 1e-7 trust gate. Slightly
    # weaker KL, longer warmup, stronger z injection, and an explicit
    # utilization hinge target the binding stochastic_latent_utilization fail.
    kl_coef=5e-4,
    kl_warmup_frac=0.50,    # linear beta warmup; avoids early posterior collapse
    free_bits=0.08,          # nats per latent dimension
    latent_inject_gain=2.5,  # residual scale for z at the dynamics bottleneck
    utilization_coef=0.15,   # relative hinge weight (O(1) when collapsed)
    utilization_target=5e-6, # soft floor above full-scale latent_utilization_min
    # ballless_positive_rate: refuse positive reward when predicted ball mass
    # is near zero (trust invariant uses the same interior-ball definition).
    # Round-0 full trust cleared latent utilization (5.6e-4) but ballless
    # stayed at 0.123 > 0.05 — raise the consistency weight for later rounds.
    ballless_reward_coef=3.0,
    ballless_mass_tau=0.5,   # soft mass scale; ~exp(-mass/tau) ballless weight
    head_event_balance=True,
    head_weight_clip=20.0,
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
    # Ensemble pessimism. All zero values reproduce legacy mean-reward dreams.
    reward_disagreement_coef=0.25,
    frame_disagreement_coef=5.0,
    done_disagreement_coef=1.0,
    uncertainty_continuation_coef=1.0,
    uncertainty_threshold=None,
    frame_mode="mean",       # deterministic; "sample" draws coherent members
    gae_lambda=0.95,
    ppo_clip=0.2,
    ppo_epochs=4,
    ppo_minibatches=8,
)
REDTEAM = dict(
    horizon=30, batch=128, gamma=0.97,
    ent_coef=0.02, vf_coef=0.5, lr=3e-4, grad_clip=0.5,
    reward_disagreement_coef=1.0,
    frame_disagreement_coef=10.0,
    done_disagreement_coef=2.0,
    ball_disappearance_coef=5.0,
    frame_mode="mean",
)

# ------------------------------------------------------- trust evaluation ----
# Deliberately forgiving preflight limits: smoke runs should catch broken
# checkpoints/contracts, not reject an under-trained 200-step model. Full runs
# may override individual values with evaluate_trust.py --threshold NAME=VALUE.
TRUST = dict(
    smoke=dict(
        natural_rollout_mse_max=0.35,
        policy_lockstep_mse_max=0.40,
        forced_scenario_mse_max=0.45,
        reward_mae_max=1.50,
        done_brier_max=0.55,
        ballless_positive_rate_max=0.50,
        dream_real_reward_gap_max=1.50,
        uncertainty_rank_corr_min=-1.0,
    ),
    local=dict(
        natural_rollout_mse_max=0.18,
        policy_lockstep_mse_max=0.22,
        forced_scenario_mse_max=0.28,
        reward_mae_max=0.55,
        done_brier_max=0.25,
        ballless_positive_rate_max=0.10,
        dream_real_reward_gap_max=0.45,
        uncertainty_rank_corr_min=0.05,
    ),
    full=dict(
        natural_rollout_mse_max=0.12,
        policy_lockstep_mse_max=0.16,
        forced_scenario_mse_max=0.20,
        reward_mae_max=0.35,
        done_brier_max=0.18,
        ballless_positive_rate_max=0.05,
        dream_real_reward_gap_max=0.30,
        uncertainty_rank_corr_min=0.10,
    ),
)

# Applied only when at least one stochastic WM is evaluated. Deterministic
# reports retain the established gates exactly.
STOCHASTIC_TRUST = dict(
    smoke=dict(
        prior_std_min=1e-4,
        prior_sample_frame_mse_min=1e-8,
        latent_utilization_min=1e-8,
        serve_direction_coverage_min=0.0,
        posterior_prior_kl_min=1e-6,
    ),
    local=dict(
        prior_std_min=1e-3,
        prior_sample_frame_mse_min=1e-7,
        latent_utilization_min=1e-7,
        serve_direction_coverage_min=1.0,
        posterior_prior_kl_min=1e-4,
    ),
    full=dict(
        prior_std_min=1e-3,
        prior_sample_frame_mse_min=1e-7,
        latent_utilization_min=1e-7,
        serve_direction_coverage_min=1.0,
        posterior_prior_kl_min=1e-4,
    ),
)

ROUND_TWO = dict(
    policy_seeds=(42, 314, 2718),
    acquisition_seeds=(1103, 2207, 3301, 4409, 5519),
    wm_seeds=(42, 314, 2718),
    development_seeds=(7001, 7003, 7013),
    final_evaluation_seeds=(104729, 130363, 155921),
    development_episodes_per_seed=100,
    final_episodes_per_seed=200,
    development_promotion_win_rate=0.80,
    minimum_win_rate=0.12,
    target_win_rate=0.20,
    max_reward_gap=0.00499,
    eval_episodes=200,
)

CAMPAIGN = dict(
    max_rounds=5,
    max_stage_retries=2,
    target_final_win_rate=0.80,
    # Fixed acquisition recipes selected only from development trust failures.
    acquisition_mix_default=(
        "tracker=.10,rare=.25,redteam=.25,stochastic=.30,natural=.10"),
    acquisition_mix_calibration=(
        "tracker=.05,rare=.40,redteam=.20,stochastic=.25,natural=.10"),
    acquisition_mix_collapse=(
        "tracker=.05,rare=.20,redteam=.20,stochastic=.45,natural=.10"),
    # Prior collapse-only mix raised stochastic share and made ballless worse
    # without lifting latent utilization. Prefer rare miss/score context plus
    # moderate stochastic pressure when BOTH binding gates fail.
    acquisition_mix_binding_repair=(
        "tracker=.10,rare=.35,redteam=.20,stochastic=.25,natural=.10"),
    acquisition_mix_uncertainty=(
        "tracker=.05,rare=.20,redteam=.40,stochastic=.25,natural=.10"),
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
    collect=1_800, wm_train=10_800, wm_eval=1_800, ablation=5_400,
    ppo=5_400, wm_v2=3_600, dream=14_400, transfer=1_800,
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
