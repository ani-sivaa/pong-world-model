import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import config
from agents.policy import PolicyNet
from agents.train_dream import (
    clipped_policy_loss,
    compute_gae,
)
from scripts.collect_adaptive import enrich_terminal_context
from scripts.evaluate_trust import stochastic_collapse_gates
from scripts.promote_round_two import aggregate_reports
from scripts.run_honest_campaign import acquisition_mix, round_plan
from scripts.run_round_two import build_plan
from scripts.select_candidate import select_candidate
from scripts.evaluate_panel import wilson_interval
from wm.data import EVENT_CODES
from wm.model import (
    StochasticWorldModel,
    WorldModel,
    WorldModelEnsemble,
    load_wm,
)
from wm.train import event_balanced_weights


class TerminalContextTest(unittest.TestCase):
    def test_context_stays_real_history_and_does_not_cross_done(self):
        events = np.zeros(12, np.uint8)
        events[8] = EVENT_CODES["concede"]
        priorities = np.ones(12, np.float32)
        dones = np.zeros(12, bool)
        dones[5] = True
        context, anchors = enrich_terminal_context(
            events, priorities, dones,
            [{"start": 0, "end": 12}], radius=8, decay=0.5)
        self.assertEqual(anchors.tolist(), [[8, EVENT_CODES["concede"]]])
        self.assertEqual(events[7], EVENT_CODES["miss"])
        self.assertTrue(np.all(context[:6] == 0))
        self.assertTrue(np.all(context[6:9] != 0))
        self.assertGreater(priorities[8], priorities[7])


class StochasticRepairTest(unittest.TestCase):
    @staticmethod
    def model():
        cfg = dict(config.WM, base_channels=8, act_embed=8, latent_dim=4)
        return StochasticWorldModel(with_heads=True, cfg=cfg).eval()

    def test_latent_changes_frames_not_reward_or_done_heads(self):
        model = self.model()
        stack = torch.rand(2, 4, 64, 64)
        action = torch.tensor([0, 2])
        with torch.no_grad():
            first = model(
                stack, action, latent_mode="sample",
                generator=torch.Generator().manual_seed(1))
            second = model(
                stack, action, latent_mode="sample",
                generator=torch.Generator().manual_seed(2))
        self.assertFalse(torch.allclose(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_event_balancing_equalizes_class_mass(self):
        events = torch.tensor([[0], [0], [0], [3]])
        alive = torch.ones(4, 1)
        weights = event_balanced_weights(events, alive)
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3]), places=5)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)

    def test_legacy_stochastic_checkpoint_preserves_latent_heads(self):
        cfg = dict(
            config.WM, base_channels=8, act_embed=8, latent_dim=4,
            stochastic_heads_deterministic=False)
        legacy = StochasticWorldModel(with_heads=True, cfg=cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save({
                "model": legacy.state_dict(),
                "model_type": "stochastic",
                "config": {
                    "base_channels": 8, "act_embed": 8, "latent_dim": 4,
                    "model_type": "stochastic",
                },
            }, path)
            loaded = load_wm(path, torch.device("cpu"), with_heads=True)
        self.assertFalse(loaded.deterministic_heads)

    def test_collapse_and_unavailable_diagnostics_fail_closed(self):
        collapsed = {
            "prior_std": 0.0,
            "prior_sample_frame_mse": 0.0,
            "latent_utilization": 0.0,
            "serve_direction_coverage": None,
            "posterior_prior_kl": 0.0,
        }
        gates, passed, _ = stochastic_collapse_gates(collapsed, "full")
        self.assertFalse(passed)
        self.assertFalse(gates["serve_direction_coverage"]["available"])


class ImaginedPPOTest(unittest.TestCase):
    def test_gae_matches_one_step_return(self):
        rewards = torch.tensor([[1.0], [2.0]])
        values = torch.zeros_like(rewards)
        continuations = torch.ones_like(rewards)
        advantages, returns = compute_gae(
            rewards, values, continuations, torch.tensor([3.0]),
            gamma=1.0, gae_lambda=1.0)
        self.assertTrue(torch.equal(returns[:, 0], torch.tensor([6.0, 5.0])))
        self.assertTrue(torch.equal(advantages, returns))

    def test_ppo_backward_updates_policy_only(self):
        cfg = dict(config.WM, base_channels=8, act_embed=8)
        wm = WorldModelEnsemble([WorldModel(with_heads=True, cfg=cfg)]).eval()
        for parameter in wm.parameters():
            parameter.requires_grad_(False)
        policy = PolicyNet()
        stack = torch.zeros(2, 4, 64, 64)
        logits, _ = policy(stack)
        actions = torch.tensor([0, 1])
        old_logp = torch.distributions.Categorical(
            logits=logits.detach()).log_prob(actions)
        with torch.no_grad():
            next_stack, _ = wm.rollout_step(stack, actions)
        new_logits, values = policy(next_stack)
        new_logp = torch.distributions.Categorical(
            logits=new_logits).log_prob(actions)
        loss = clipped_policy_loss(
            new_logp, old_logp, torch.ones(2), torch.ones(2), 0.2)
        loss = loss + values.square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in policy.parameters()))
        self.assertTrue(all(p.grad is None for p in wm.parameters()))


class RoundTwoOrchestrationTest(unittest.TestCase):
    def test_dry_plan_is_gated_has_three_seeds_and_no_export(self):
        args = SimpleNamespace(
            backend="local", scale="smoke", tag="test-round",
            base_wm=["old0.pt", "old1.pt"], redteam="red.pt",
            stochastic_policy="stochastic.pt", transitions=64, steps=1,
            updates=1)
        stages = build_plan(args)
        names = [name for name, _, _ in stages]
        self.assertLess(names.index("trust-before-policy"),
                        names.index("train-policy-0"))
        self.assertEqual(sum(name.startswith("train-policy-") for name in names), 3)
        self.assertNotIn("export", " ".join(names).lower())
        commands = [" ".join(command) for _, command, _ in stages]
        self.assertEqual(sum("--episodes 200" in item for item in commands), 3)

    def test_promotion_requires_every_seed_and_gate(self):
        gate = {"available": True, "pass": True}
        trust = {
            "overall_pass": True,
            "gates": {"x": gate},
            "measurements": {"current_policy_lockstep": {
                "dream_real_reward_gap": {"available": True, "value": 0.001}}},
        }
        transfer = {"dream": {"win_rate": 0.13}}
        accepted = aggregate_reports(
            [trust] * 3, [transfer] * 3, 0.12, 0.20, 0.005)
        self.assertTrue(accepted["accepted"])
        broken = dict(trust, gates={"x": {"available": False, "pass": None}})
        rejected = aggregate_reports(
            [trust, broken, trust], [transfer] * 3, 0.12, 0.20, 0.005)
        self.assertFalse(rejected["accepted"])
        self.assertIsNone(rejected["export_policy"])

    def test_campaign_separates_development_and_final_panels(self):
        args = SimpleNamespace(
            tag="campaign-test", redteam="/vol/checkpoints/red.pt",
            stochastic_policy="/vol/checkpoints/stochastic.pt",
            base_wm=["/vol/checkpoints/old.pt"], transitions=100,
            steps=1, updates=1)
        stages, _ = round_plan(args, 0, acquisition_mix(None))
        joined = "\n".join(" ".join(command) for _, command, _, _ in stages)
        self.assertIn("--purpose development", joined)
        self.assertNotIn("--purpose final", joined)
        self.assertEqual(
            sum(name.startswith("development-") for name, *_ in stages), 3)

    def test_candidate_selection_never_reads_final_results(self):
        trust = {"overall_pass": True,
                 "gates": {"all": {"available": True, "pass": True}}}
        developments = [
            {"mean_win_rate": 0.81, "pooled_win_rate_ci95": [0.7, 0.9]},
            {"mean_win_rate": 0.85, "pooled_win_rate_ci95": [0.8, 0.9]},
            {"mean_win_rate": 0.85, "pooled_win_rate_ci95": [0.8, 0.9]},
        ]
        selected = select_candidate(
            ["a.pt", "b.pt", "c.pt"], [trust] * 3, developments, 0.8)
        self.assertTrue(selected["selection_uses_final_results"] is False)
        self.assertEqual(selected["selected_candidate_index"], 1)

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(160, 200)
        self.assertLess(low, 0.8)
        self.assertGreater(high, 0.8)


if __name__ == "__main__":
    unittest.main()
