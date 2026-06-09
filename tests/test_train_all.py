import json
import tempfile
import unittest
from pathlib import Path

from tools import train_all


class TrainAllTest(unittest.TestCase):
    def test_brain_output_paths(self):
        self.assertEqual(train_all.brain_output("dqn"), train_all.MAIN_MANIFEST)
        self.assertEqual(train_all.brain_output("ppo"), Path("js/brains/ppo.json"))
        self.assertEqual(train_all.brain_output("qrdqn"), Path("js/brains/qr-dqn.json"))
        self.assertEqual(train_all.brain_output("maskable-ppo"), Path("js/brains/maskable-ppo.json"))

    def test_checkpoint_dir_namespaced_per_technique(self):
        self.assertEqual(train_all.checkpoint_dir("dqn"), Path(".training-checkpoints/galagai-balanced-v16"))
        self.assertEqual(train_all.checkpoint_dir("ppo"), Path(".training-checkpoints/galagai-balanced-v16-ppo"))

    def test_brain_manifest_url_relative_to_main(self):
        self.assertEqual(train_all.brain_manifest_url("ppo"), "brains/ppo.json")
        self.assertEqual(train_all.brain_manifest_url("qrdqn"), "brains/qr-dqn.json")

    def test_publish_command_forwards_algorithm_and_paths(self):
        command = train_all.publish_command("ppo", target_rounds=4, shared_args=["--train-workers", "1"], python="python")
        self.assertIn("--algorithm", command)
        self.assertEqual(command[command.index("--algorithm") + 1], "ppo")
        self.assertEqual(command[command.index("--out") + 1], "js/brains/ppo.json")
        self.assertIn("--no-push", command)
        self.assertIn("--no-pages", command)
        self.assertEqual(command[command.index("--target-rounds") + 1], "4")

    def test_assemble_brains_index_lists_only_existing_non_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main = root / "galagai-model.json"
            main.write_text(json.dumps({"version": 16, "technique": "dqn"}), encoding="utf-8")
            brains_dir = root / "brains"
            brains_dir.mkdir()
            (brains_dir / "ppo.json").write_text("{}", encoding="utf-8")  # exists
            # qr-dqn intentionally NOT created -> must be skipped

            orig_main, orig_brains = train_all.MAIN_MANIFEST, train_all.BRAINS_DIR
            try:
                train_all.MAIN_MANIFEST = main
                train_all.BRAINS_DIR = brains_dir
                index = train_all.assemble_brains_index(main, ["dqn", "ppo", "qrdqn"])
            finally:
                train_all.MAIN_MANIFEST, train_all.BRAINS_DIR = orig_main, orig_brains

            self.assertIn("ppo", index)
            self.assertNotIn("qr-dqn", index)  # file missing -> skipped
            self.assertNotIn("dqn", index)  # default brain is implicit
            self.assertEqual(index["ppo"]["manifest"], "brains/ppo.json")
            self.assertEqual(index["ppo"]["algorithm"], "stable-baselines3-ppo")
            # Persisted back into the manifest.
            written = json.loads(main.read_text(encoding="utf-8"))
            self.assertEqual(written["brains"], index)


if __name__ == "__main__":
    unittest.main()
