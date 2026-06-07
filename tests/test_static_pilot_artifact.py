import json
import tempfile
import unittest
from pathlib import Path

from tools import train_static_pilot


class StaticPilotArtifactTest(unittest.TestCase):
    def test_environment_penalizes_drop_spam(self):
        env = train_static_pilot.HeadlessGalagai(seed=12, max_steps=20)

        _, _, first_enemy_reward, _, first_info = env.step(3, 1)
        _, _, second_enemy_reward, _, second_info = env.step(3, 1)

        self.assertTrue(first_info["events"].valid_drop)
        self.assertFalse(first_info["events"].invalid_drop)
        self.assertTrue(second_info["events"].invalid_drop)
        self.assertLess(second_enemy_reward, first_enemy_reward)
        self.assertEqual(second_info["invalidDrops"], 1)

    def test_write_model_includes_exported_pilot_and_enemy_networks(self):
        pilot_model, enemy_model, self_play = train_static_pilot.train_self_play(
            seed=21,
            rounds=2,
            timesteps_per_round=64,
            eval_episodes=2,
            max_steps=80,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            train_static_pilot.write_model(path, pilot_model, enemy_model, self_play)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["version"], 4)
        self.assertEqual(payload["algorithm"], "stable-baselines3-dqn")
        self.assertEqual(payload["actions"], train_static_pilot.PILOT_ACTIONS)
        self.assertEqual(payload["features"], train_static_pilot.FEATURES)
        self.assertEqual(payload["enemies"]["actions"], train_static_pilot.ENEMY_ACTIONS)
        self.assertGreaterEqual(len(payload["network"]["layers"]), 2)
        self.assertGreaterEqual(len(payload["enemies"]["network"]["layers"]), 2)
        self.assertEqual(payload["metrics"]["selfPlay"]["latest"]["round"], 2)
        self.assertIn("invalidDropRate", payload["metrics"])


if __name__ == "__main__":
    unittest.main()
