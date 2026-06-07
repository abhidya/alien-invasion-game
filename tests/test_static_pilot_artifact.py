import json
import tempfile
import unittest
from pathlib import Path

from tools import train_static_pilot


class StaticPilotArtifactTest(unittest.TestCase):
    def test_write_model_includes_self_play_metrics(self):
        training = train_static_pilot.build_dataset(seed=11, samples=96)
        evaluation = train_static_pilot.build_dataset(seed=12, samples=64)
        weights = train_static_pilot.train(training, epochs=10, lr=0.2, l2=0.001)
        train_acc = train_static_pilot.accuracy(weights, training)
        eval_acc = train_static_pilot.accuracy(weights, evaluation)
        matrix = train_static_pilot.confusion(weights, evaluation)
        self_play = train_static_pilot.alternating_curriculum(seed=13, eval_acc=eval_acc, rounds=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            train_static_pilot.write_model(path, weights, train_acc, eval_acc, matrix, self_play)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["version"], 2)
        self.assertEqual(payload["actions"], train_static_pilot.ACTIONS)
        self.assertEqual(payload["features"], train_static_pilot.FEATURES)
        self.assertIn("selfPlay", payload["metrics"])
        self.assertEqual(payload["metrics"]["selfPlay"]["latest"]["round"], 4)


if __name__ == "__main__":
    unittest.main()
