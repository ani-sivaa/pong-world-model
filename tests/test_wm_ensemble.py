import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np
import torch

from wm.data import TransitionData
from wm.model import (StochasticWorldModel, WorldModel, WorldModelEnsemble,
                      load_wm, load_wm_ensemble)
from wm.train import compute_losses
from agents.train_dream import pessimistic_targets


class WorldModelEnsembleTest(unittest.TestCase):
    @staticmethod
    def stochastic_cfg(latent_dim=4):
        import config
        return dict(config.WM, base_channels=8, act_embed=8,
                    latent_dim=latent_dim)

    def test_old_checkpoint_and_ensemble_statistics(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            # Legacy headless checkpoints remain loadable through load_wm.
            old = WorldModel(with_heads=False)
            old_path = tmp / "old.pt"
            torch.save({"model": old.state_dict(), "step": 1}, old_path)
            loaded = load_wm(old_path, device, with_heads=False)
            self.assertIsInstance(loaded, WorldModel)

            first = WorldModel(with_heads=True)
            second = WorldModel(with_heads=True)
            second.load_state_dict(first.state_dict())
            with torch.no_grad():
                second.out.bias.add_(0.5)
                second.reward_head[-1].bias.add_(1.0)
                second.done_head[-1].bias.add_(0.5)
            paths = [tmp / "member0.pt", tmp / "member1.pt"]
            torch.save({"model": first.state_dict()}, paths[0])
            torch.save({"model": second.state_dict()}, paths[1])

            ensemble = load_wm_ensemble(paths, device, with_heads=True).eval()
            stack = torch.zeros(2, 4, 64, 64)
            action = torch.tensor([0, 2])
            with torch.no_grad():
                prediction = ensemble(stack, action)
                next_stack, repeated = ensemble.rollout_step(
                    stack, action, frame_mode="mean")

            self.assertEqual(prediction.member_frames.shape, (2, 2, 1, 64, 64))
            self.assertEqual(prediction.member_rewards.shape, (2, 2))
            self.assertEqual(prediction.mean_done_prob.shape, (2,))
            self.assertEqual(next_stack.shape, stack.shape)
            self.assertTrue(torch.allclose(next_stack[:, -1:], repeated.mean_frame))
            self.assertTrue(torch.all(prediction.frame_mse > 0))
            self.assertTrue(torch.all(prediction.reward_std > 0))
            self.assertTrue(torch.all(prediction.done_disagreement > 0))

    def test_single_member_has_zero_disagreement(self):
        model = WorldModel(with_heads=True).eval()
        from wm.model import WorldModelEnsemble

        ensemble = WorldModelEnsemble([model])
        with torch.no_grad():
            prediction = ensemble.predict(
                torch.zeros(1, 4, 64, 64), torch.zeros(1, dtype=torch.long))
        self.assertEqual(float(prediction.frame_mse), 0.0)
        self.assertEqual(float(prediction.reward_variance), 0.0)
        self.assertEqual(float(prediction.done_disagreement), 0.0)

    def test_stochastic_checkpoint_auto_load_and_prior_modes(self):
        device = torch.device("cpu")
        model = StochasticWorldModel(
            with_heads=True, cfg=self.stochastic_cfg()).eval()
        stack = torch.zeros(2, 4, 64, 64)
        action = torch.tensor([0, 2])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stochastic.pt"
            torch.save({
                "model": model.state_dict(),
                "model_type": "stochastic",
                "config": dict(model.model_config, model_type="stochastic"),
            }, path)
            loaded = load_wm(path, device, with_heads=True).eval()
            self.assertIsInstance(loaded, StochasticWorldModel)
            with torch.no_grad():
                mean_a = loaded(stack, action, latent_mode="mean")[0]
                mean_b = loaded(stack, action, latent_mode="mean")[0]
                generator = torch.Generator().manual_seed(9)
                sample_a = loaded(
                    stack, action, latent_mode="sample", generator=generator)[0]
                sample_b = loaded(
                    stack, action, latent_mode="sample", generator=generator)[0]
                seeded_a = loaded(
                    stack, action, latent_mode="sample",
                    generator=torch.Generator().manual_seed(77))[0]
                seeded_b = loaded(
                    stack, action, latent_mode="sample",
                    generator=torch.Generator().manual_seed(77))[0]
            self.assertTrue(torch.equal(mean_a, mean_b))
            self.assertFalse(torch.allclose(sample_a, sample_b))
            self.assertTrue(torch.equal(seeded_a, seeded_b))

    def test_stochastic_kl_is_finite_and_backpropagates(self):
        model = StochasticWorldModel(
            with_heads=False, cfg=self.stochastic_cfg())
        batch_size = 2
        batch = (
            np.zeros((batch_size, 4, 64, 64), np.uint8),
            np.zeros((batch_size, 1), np.int64),
            np.zeros((batch_size, 1, 64, 64), np.uint8),
            np.zeros((batch_size, 1), np.float32),
            np.zeros((batch_size, 1), np.float32),
            np.ones((batch_size, 1), np.float32),
        )
        total, _, _, _, kl, _, _ = compute_losses(
            model, batch, torch.device("cpu"), False, 1.0,
            kl_coef=0.1, free_bits=0.0, return_kl=True)
        total.backward()
        self.assertTrue(torch.isfinite(kl))
        self.assertGreaterEqual(float(kl.detach()), 0.0)
        self.assertIsNotNone(model.posterior_stats.weight.grad)
        self.assertTrue(torch.isfinite(model.posterior_stats.weight.grad).all())

    def test_stochastic_ensemble_epistemic_metrics_are_deterministic(self):
        model = StochasticWorldModel(
            with_heads=True, cfg=self.stochastic_cfg()).eval()
        ensemble = WorldModelEnsemble([model])
        stack = torch.zeros(2, 4, 64, 64)
        action = torch.zeros(2, dtype=torch.long)
        with torch.no_grad():
            first = ensemble.predict(stack, action)
            generator = torch.Generator().manual_seed(123)
            _, sampled = ensemble.rollout_step(
                stack, action, latent_mode="sample", generator=generator)
            second = ensemble.predict(stack, action)
        self.assertTrue(torch.equal(first.member_frames, second.member_frames))
        self.assertTrue(torch.equal(first.frame_mse, sampled.frame_mse))
        self.assertEqual(float(sampled.frame_mse.max()), 0.0)
        self.assertGreater(float(sampled.aleatoric_frame_mse.max()), 0.0)

    def test_seeded_window_bootstrap_is_reproducible_and_diverse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            count = 100
            frames = np.zeros((count, 64, 64), np.uint8)
            frames[:, 0, 0] = np.arange(count, dtype=np.uint8)
            np.save(root / "frames.npy", frames)
            np.save(root / "actions.npy", np.arange(count, dtype=np.uint8) % 3)
            np.save(root / "rewards.npy", np.zeros(count, np.float32))
            dones = np.zeros(count, bool)
            dones[19::20] = True
            np.save(root / "dones.npy", dones)
            (root / "meta.json").write_text(json.dumps({"T": count}))

            def indices(seed):
                data = TransitionData(
                    root, seed=seed, bootstrap=True, bootstrap_frac=1.0)
                _, info = data.sample(32, 3, return_info=True)
                return info["indices"]

            self.assertTrue(np.array_equal(indices(7), indices(7)))
            self.assertFalse(np.array_equal(indices(7), indices(8)))

    def test_pessimism_penalty_discount_and_threshold(self):
        prediction = SimpleNamespace(
            reward_std=torch.tensor([0.2, 0.1]),
            frame_mse=torch.tensor([0.1, 0.0]),
            done_disagreement=torch.tensor([0.1, 0.0]),
            mean_reward=torch.tensor([1.0, 0.5]),
            mean_done_prob=torch.tensor([0.25, 0.25]),
        )
        reward, continuation, uncertainty, penalty = pessimistic_targets(
            prediction, reward_coef=1.0, frame_coef=2.0, done_coef=3.0,
            continuation_coef=1.0, threshold=0.2)
        self.assertTrue(torch.allclose(penalty, torch.tensor([0.7, 0.1])))
        self.assertTrue(torch.allclose(reward, torch.tensor([0.3, 0.4])))
        self.assertTrue(torch.allclose(uncertainty, torch.tensor([0.4, 0.1])))
        self.assertEqual(float(continuation[0]), 0.0)
        self.assertGreater(float(continuation[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
