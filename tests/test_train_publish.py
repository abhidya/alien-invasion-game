import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from tools import train_publish


class TrainPublishTest(unittest.TestCase):
    def _args(self, checkpoint_dir: Path) -> Namespace:
        return Namespace(
            checkpoint_dir=checkpoint_dir,
            min_balanced_rounds=12,
            balance_tolerance=0.2,
            balance_patience=3,
            balance_min_win_rate=0.25,
            phase_timesteps=900,
            max_phase_iterations=4,
            eval_episodes=10,
            max_steps=360,
            dominance_threshold=0.65,
            train_workers=4,
            eval_workers=4,
            curriculum_waves=3,
            candidate_spawns=2,
            checkpoint_retention="tiered",
            keep_latest_versions=12,
            model=Path("js/galagai-model.json"),
            no_resume=True,
            no_progress=False,
        )

    def test_default_checkpoint_dir_uses_current_schema_line(self):
        self.assertEqual(train_publish.DEFAULT_CHECKPOINT_DIR, Path(".training-checkpoints/galagai-balanced-v14"))

    def test_add_rounds_command_requires_new_rounds_after_resume(self):
        args = self._args(Path(".training-checkpoints/galagai-balanced-v14"))

        command = train_publish.build_train_command(args, target_rounds=300, required_new_balanced_rounds=43)

        required_index = command.index("--required-new-balanced-rounds")
        self.assertEqual(command[required_index + 1], "43")

    def test_interrupt_exports_recovered_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "state.json").write_text(
                json.dumps({"rounds": [{}, {}], "roundNumber": 2}),
                encoding="utf-8",
            )
            args = self._args(checkpoint_dir)
            calls = []

            def fake_run(command, **_kwargs):
                calls.append(command)
                if len(calls) == 1:
                    raise subprocess.CalledProcessError(-2, command)
                return ""

            with mock.patch.object(train_publish, "run", side_effect=fake_run):
                interrupted = train_publish.run_training(args, target_rounds=5, current_rounds=1)

        self.assertTrue(interrupted)
        self.assertEqual(len(calls), 2)
        self.assertIn("--resume", calls[1])
        self.assertIn("--no-progress", calls[1])
        self.assertIn("--curriculum-waves", calls[1])
        self.assertIn("--candidate-spawns", calls[1])
        target_index = calls[1].index("--balanced-rounds")
        self.assertEqual(calls[1][target_index + 1], "2")
        required_index = calls[1].index("--required-new-balanced-rounds")
        self.assertEqual(calls[1][required_index + 1], "0")

    def test_public_manifest_check_retries_until_pages_updates(self):
        expected = {
            "version": 14,
            "pilotVersions": 18,
            "enemyVersions": 1,
            "latestPilot": "galagai-models/pilot-v023.json",
            "latestEnemy": "galagai-models/enemies-v001.json",
        }
        stale = json.dumps(
            {
                "version": 10,
                "versions": {"pilot": [{}], "enemies": [{}]},
                "networkRef": "old-pilot.json",
                "enemies": {"networkRef": "old-enemy.json"},
            }
        )
        fresh = json.dumps(
            {
                "version": 14,
                "versions": {"pilot": [{} for _ in range(18)], "enemies": [{}]},
                "networkRef": "galagai-models/pilot-v023.json",
                "enemies": {"networkRef": "galagai-models/enemies-v001.json"},
            }
        )
        responses = [stale, fresh]

        with mock.patch.object(train_publish, "run", side_effect=lambda *_args, **_kwargs: responses.pop(0)):
            with mock.patch.object(train_publish.time, "sleep") as sleep:
                train_publish.public_manifest_check(expected, attempts=2, delay_seconds=0.01)

        sleep.assert_called_once_with(0.01)


if __name__ == "__main__":
    unittest.main()
