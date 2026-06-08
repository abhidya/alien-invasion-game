"""Recommendation B, Python side: pin the trainer's observation encoder.

Replays the committed golden scenes through the real Python encoder and asserts
the observation is unchanged. This is the regression guard: editing encode_frame
or scalar_features in a way that shifts the layout fails here, before a retrained
agent inherits a different observation than the browser produces.

Regenerate intentionally with: ./.venv-rl/bin/python -m tools.gen_encoder_golden
"""

import json
import unittest
from pathlib import Path

from tools.gen_encoder_golden import build_env_from_scene

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "encoder_golden.json"


class EncoderContractTest(unittest.TestCase):
    def setUp(self):
        with open(FIXTURE, encoding="utf-8") as handle:
            self.golden = json.load(handle)

    def test_python_encoder_matches_golden(self):
        scenes = self.golden["scenes"]
        observations = self.golden["observations"]
        self.assertEqual(len(scenes), len(observations))
        self.assertTrue(scenes, "fixture has no scenes")
        for scene, expected in zip(scenes, observations):
            env, controlled = build_env_from_scene(scene)
            obs = env.observation(controlled).tolist()
            self.assertEqual(len(obs), len(expected), scene["name"])
            for i, (got, want) in enumerate(zip(obs, expected)):
                self.assertAlmostEqual(
                    got, want, places=5, msg=f"{scene['name']} index {i}"
                )

    def test_golden_observation_length_is_frame_plus_scalars(self):
        # 8 channels x 28 rows x 48 cols + 6 scalars.
        self.assertEqual(len(self.golden["observations"][0]), 8 * 28 * 48 + 6)


if __name__ == "__main__":
    unittest.main()
