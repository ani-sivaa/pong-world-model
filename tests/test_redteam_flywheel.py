import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import config
from agents.policy import PolicyNet
from agents.train_redteam import redteam_score
from scripts.collect_adaptive import _priority_bonus, collect
from wm.data import EVENT_CODES


class RedTeamObjectiveTest(unittest.TestCase):
    def test_objective_favors_disagreement_and_ball_disappearance(self):
        previous = torch.zeros(1, 1, 64, 64)
        previous[0, 0, 20, 20:22] = 1
        healthy_members = previous.unsqueeze(0).repeat(2, 1, 1, 1, 1)
        failure_members = torch.zeros_like(healthy_members)

        def prediction(frames, disagreement):
            return SimpleNamespace(
                member_frames=frames,
                reward_std=torch.tensor([disagreement]),
                frame_mse=torch.tensor([disagreement]),
                done_disagreement=torch.tensor([disagreement]),
            )

        healthy, _ = redteam_score(
            prediction(healthy_members, 0.0), previous, 1, 1, 1, 5)
        failing, components = redteam_score(
            prediction(failure_members, 0.25), previous, 1, 1, 1, 5)
        self.assertEqual(float(healthy), 0.0)
        self.assertGreater(float(failing), float(healthy))
        self.assertEqual(float(components["ball_disappearance"]), 1.0)

    def test_priority_includes_ensemble_and_real_mismatches(self):
        prediction = {
            "frame": np.zeros((1, 64, 64), np.float32),
            "reward": np.zeros(1, np.float32),
            "done": np.zeros(1, np.float32),
            "frame_disagreement": np.ones(1, np.float32),
            "reward_disagreement": np.ones(1, np.float32),
            "done_disagreement": np.ones(1, np.float32),
        }
        bonus = _priority_bonus(
            prediction, np.full((1, 64, 64), 255, np.uint8),
            np.ones(1, np.float32), np.ones(1, bool))
        expected = (3 * config.FLYWHEEL["disagreement_weight"]
                    + config.FLYWHEEL["frame_mismatch_weight"]
                    + config.FLYWHEEL["reward_mismatch_weight"]
                    + config.FLYWHEEL["done_mismatch_weight"])
        self.assertAlmostEqual(float(bonus[0]), expected)


class AdaptiveRedTeamCollectionTest(unittest.TestCase):
    def test_redteam_actions_drive_real_boundary_safe_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_path = root / "redteam.pt"
            policy = PolicyNet()
            with torch.no_grad():
                policy.pi.weight.zero_()
                policy.pi.bias.copy_(torch.tensor([-100.0, -100.0, 100.0]))
            torch.save({"model": policy.state_dict()}, policy_path)
            args = SimpleNamespace(
                scale="smoke", out=str(root / "data"), transitions=24, seed=17,
                mix="redteam=1", baseline_checkpoint=str(root / "missing-base.pt"),
                dream_checkpoint=str(root / "missing-dream.pt"),
                redteam_checkpoint=str(policy_path),
                world_model_checkpoint=None,
            )
            out, meta = collect(args)
            frames = np.load(out / "frames.npy")
            actions = np.load(out / "actions.npy")
            dones = np.load(out / "dones.npy")
            events = np.load(out / "events.npy")

            self.assertTrue(meta["real_frames_only"])
            self.assertEqual(meta["segments"][0]["actual_policy"], "redteam")
            self.assertTrue(np.all(actions == 2))
            self.assertTrue(np.isin(frames, [0, 255]).all())
            for stream in meta["streams"]:
                boundary = stream["end"] - 1
                self.assertTrue(dones[boundary])
                self.assertEqual(events[boundary], EVENT_CODES["boundary"])
                for index in range(stream["start"] + config.FRAME_STACK - 1,
                                   stream["end"]):
                    self.assertFalse(dones[index - 3:index].any())


if __name__ == "__main__":
    unittest.main()
