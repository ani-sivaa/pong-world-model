import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import config
from scripts.evaluate_trust import (
    aggregate_gates,
    binary_auroc,
    calibration_metrics,
    rank_correlation,
)
from wm.model import WorldModel


REPO = Path(__file__).resolve().parents[1]


class TrustMetricTest(unittest.TestCase):
    def test_rank_metrics_and_gate_aggregation(self):
        self.assertAlmostEqual(rank_correlation([0, 1, 2], [0, 2, 4]), 1.0)
        self.assertAlmostEqual(binary_auroc([0.1, 0.9, 0.2, 0.8],
                                            [False, True, False, True]), 1.0)
        self.assertIsNone(binary_auroc([0.1, 0.2], [True, True]))

        metrics = {
            "error": {"available": True, "value": 0.2},
            "correlation": {"available": False, "reason": "constant"},
        }
        gates, overall = aggregate_gates(
            metrics, {"error_max": 0.3, "correlation_min": 0.1})
        self.assertTrue(overall)
        self.assertTrue(gates["error"]["pass"])
        self.assertIsNone(gates["correlation"]["pass"])

        _, failed = aggregate_gates(metrics, {"error_max": 0.1})
        self.assertFalse(failed)

    def test_absent_calibration_classes_are_unavailable(self):
        prediction = {
            "frame": np.zeros((3, 64, 64), np.float32),
            "reward": np.zeros(3, np.float32),
            "done": np.zeros(3, np.float32),
            "uncertainty": np.arange(3, dtype=np.float32),
        }
        targets = (
            np.zeros((3, 64, 64), np.float32),
            np.zeros(3, np.float32),
            np.zeros(3, np.float32),
        )
        metrics = calibration_metrics(
            prediction, targets, np.zeros(3, dtype=np.uint8))
        self.assertFalse(metrics["reward_nonzero_mae"]["available"])
        self.assertFalse(metrics["done_positive_brier"]["available"])
        self.assertFalse(metrics["event_hit_reward_mae"]["available"])
        self.assertTrue(metrics["reward_zero_mae"]["available"])


class TrustCliSmokeTest(unittest.TestCase):
    def test_tiny_checkpoint_and_data_write_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            count = 48
            frames = np.zeros((count, 64, 64), np.uint8)
            frames[:, 0, :] = 255
            frames[:, -1, :] = 255
            frames[:, 30:32, 30:32] = 255
            np.save(data_dir / "frames.npy", frames)
            np.save(data_dir / "actions.npy",
                    np.arange(count, dtype=np.uint8) % config.N_ACTIONS)
            np.save(data_dir / "rewards.npy", np.zeros(count, np.float32))
            dones = np.zeros(count, bool)
            dones[11::12] = True
            np.save(data_dir / "dones.npy", dones)
            np.save(data_dir / "events.npy", np.zeros(count, np.uint8))
            (data_dir / "meta.json").write_text(json.dumps({"T": count}))

            model_cfg = dict(config.WM, base_channels=8, act_embed=8)
            model = WorldModel(with_heads=True, cfg=model_cfg)
            checkpoint = root / "wm.pt"
            torch.save({
                "model": model.state_dict(),
                "config": {"base_channels": 8, "act_embed": 8},
            }, checkpoint)
            report = root / "trust.json"
            result = subprocess.run([
                sys.executable, "-m", "scripts.evaluate_trust",
                "--wm", str(checkpoint), "--data", str(data_dir),
                "--scale", "smoke", "--samples", "4", "--horizon", "2",
                "--out", str(report),
            ], cwd=REPO, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(report.read_text())
            self.assertIn("overall_pass", payload)
            self.assertIn("forced_scenarios", payload["measurements"])
            self.assertFalse(payload["measurements"]["reward_done_calibration"]
                             ["done_positive_brier"]["available"])


if __name__ == "__main__":
    unittest.main()
