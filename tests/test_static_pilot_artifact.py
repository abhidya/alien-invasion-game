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
            cycles=1,
            phase_timesteps=64,
            eval_episodes=2,
            max_steps=80,
            dominance_threshold=1.1,
            max_phase_iterations=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            train_static_pilot.write_model(path, pilot_model, enemy_model, self_play)
            payload = json.loads(path.read_text(encoding="utf-8"))
            pilot_payload = json.loads((path.parent / payload["networkRef"]).read_text(encoding="utf-8"))
            enemy_payload = json.loads((path.parent / payload["enemies"]["networkRef"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["version"], 6)
        self.assertEqual(payload["algorithm"], "stable-baselines3-dqn")
        self.assertEqual(payload["actions"], train_static_pilot.PILOT_ACTIONS)
        self.assertEqual(payload["features"], train_static_pilot.FEATURES)
        self.assertEqual(payload["enemies"]["actions"], train_static_pilot.ENEMY_ACTIONS)
        self.assertEqual(len(payload["versions"]["pilot"]), 1)
        self.assertEqual(len(payload["versions"]["enemies"]), 1)
        self.assertNotIn("network", payload["versions"]["pilot"][0])
        self.assertNotIn("network", payload["versions"]["enemies"][0])
        self.assertEqual(payload["networkRef"], payload["versions"]["pilot"][0]["url"])
        self.assertEqual(payload["enemies"]["networkRef"], payload["versions"]["enemies"][0]["url"])

        self.assertGreaterEqual(len(pilot_payload["network"]["layers"]), 2)
        self.assertGreaterEqual(len(enemy_payload["network"]["layers"]), 2)
        self.assertEqual(payload["metrics"]["selfPlay"]["latest"]["round"], 2)
        self.assertEqual(payload["metrics"]["selfPlay"]["checkpointCounts"]["pilot"], 1)
        self.assertEqual(payload["metrics"]["selfPlay"]["checkpointCounts"]["enemies"], 1)
        self.assertIn("invalidDropRate", payload["metrics"])

    def test_dominance_gate_repeats_phase_until_threshold_or_cap(self):
        _, _, self_play = train_static_pilot.train_self_play(
            seed=31,
            cycles=1,
            phase_timesteps=32,
            eval_episodes=1,
            max_steps=40,
            dominance_threshold=1.1,
            max_phase_iterations=2,
        )

        self.assertEqual([round_info["trained"] for round_info in self_play["rounds"]], ["pilot", "pilot", "enemies", "enemies"])
        self.assertEqual(len(self_play["checkpoints"]["pilot"]), 2)
        self.assertEqual(len(self_play["checkpoints"]["enemies"]), 2)
        self.assertTrue(all(not round_info["dominanceReached"] for round_info in self_play["rounds"]))

    def test_generation_target_exports_exact_count_per_side(self):
        _, _, self_play = train_static_pilot.train_self_play(
            seed=41,
            generations_per_side=2,
            phase_timesteps=32,
            eval_episodes=1,
            max_steps=40,
            dominance_threshold=1.1,
            max_phase_iterations=1,
        )

        self.assertEqual(len(self_play["checkpoints"]["pilot"]), 2)
        self.assertEqual(len(self_play["checkpoints"]["enemies"]), 2)
        self.assertEqual(self_play["generationsPerSide"], 2)


if __name__ == "__main__":
    unittest.main()
