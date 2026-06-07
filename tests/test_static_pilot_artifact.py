import json
import tempfile
import unittest
from pathlib import Path

from tools import train_static_pilot


class StaticPilotArtifactTest(unittest.TestCase):
    def test_write_model_includes_pilot_and_enemy_rl_weights(self):
        pilot, enemies, self_play = train_static_pilot.train_self_play(
            seed=21,
            rounds=2,
            episodes_per_round=8,
            eval_episodes=4,
            max_steps=35,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            train_static_pilot.write_model(path, pilot, enemies, self_play)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["version"], 3)
        self.assertEqual(payload["model"], "numpy-linear-dqn-self-play")
        self.assertEqual(payload["actions"], train_static_pilot.ACTIONS)
        self.assertEqual(payload["features"], train_static_pilot.FEATURES)
        self.assertEqual(payload["enemies"]["actions"], train_static_pilot.ENEMY_ACTIONS)
        self.assertEqual(len(payload["weights"]), len(train_static_pilot.FEATURES))
        self.assertEqual(len(payload["enemies"]["weights"]), len(train_static_pilot.FEATURES))
        self.assertEqual(payload["metrics"]["selfPlay"]["latest"]["round"], 2)


if __name__ == "__main__":
    unittest.main()
