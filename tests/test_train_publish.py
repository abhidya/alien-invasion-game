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
            replay_buffer_size=50_000,
            pilot_warmup_generations=0,
            enemy_warmup_generations=0,
            model=Path("js/galagai-model.json"),
            no_resume=True,
            no_progress=False,
            skip_tests=True,
            no_commit=True,
            no_push=True,
            no_pages=True,
            publish_interval_seconds=0.0,
        )

    def test_default_checkpoint_dir_uses_current_schema_line(self):
        self.assertEqual(train_publish.DEFAULT_CHECKPOINT_DIR, Path(".training-checkpoints/galagai-balanced-v15"))
        self.assertEqual(train_publish.EXPECTED_MODEL_SCHEMA_VERSION, 15)

    def test_add_rounds_command_requires_new_rounds_after_resume(self):
        args = self._args(Path(".training-checkpoints/galagai-balanced-v15"))

        command = train_publish.build_train_command(args, target_rounds=300, required_new_balanced_rounds=43)

        required_index = command.index("--required-new-balanced-rounds")
        self.assertEqual(command[required_index + 1], "43")

    def test_warmup_generation_flags_are_forwarded_to_trainer(self):
        args = self._args(Path(".training-checkpoints/galagai-balanced-v15"))
        args.pilot_warmup_generations = 3
        args.enemy_warmup_generations = 1

        command = train_publish.build_train_command(args, target_rounds=4, required_new_balanced_rounds=4)

        pilot_index = command.index("--pilot-warmup-generations")
        enemy_index = command.index("--enemy-warmup-generations")
        self.assertEqual(command[pilot_index + 1], "3")
        self.assertEqual(command[enemy_index + 1], "1")

    def test_interrupt_exports_recovered_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "state.json").write_text(
                json.dumps({"rounds": [{}, {}], "roundNumber": 2, "totalGenerationCounts": {"pilot": 1, "enemies": 1}}),
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

    def test_interrupt_rejects_unpublishable_pilot_only_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "state.json").write_text(
                json.dumps({"rounds": [{"trained": "pilot"}], "roundNumber": 1, "totalGenerationCounts": {"pilot": 1, "enemies": 0}}),
                encoding="utf-8",
            )
            args = self._args(checkpoint_dir)

            def fake_run(command, **_kwargs):
                raise subprocess.CalledProcessError(-2, command)

            with mock.patch.object(train_publish, "run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "nothing publishable"):
                    train_publish.run_training(args, target_rounds=5, current_rounds=0)

    def test_resume_rejects_stale_schema_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "galagai-balanced-v12"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "state.json").write_text(
                json.dumps({"schemaVersion": 12, "rounds": [{}, {}], "roundNumber": 2}),
                encoding="utf-8",
            )
            args = self._args(checkpoint_dir)
            args.no_resume = False

            with self.assertRaisesRegex(RuntimeError, "does not match current schema 15"):
                train_publish.validate_resume_checkpoint(args)

            args.no_resume = True
            train_publish.validate_resume_checkpoint(args)


if __name__ == "__main__":
    unittest.main()
