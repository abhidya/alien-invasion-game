import unittest

from tools import rl_algorithms


class RlAlgorithmsTest(unittest.TestCase):
    def test_default_is_dqn_and_keys_are_stable(self):
        self.assertEqual(rl_algorithms.DEFAULT_ALGORITHM, "dqn")
        self.assertEqual(
            rl_algorithms.algorithm_keys(),
            ["dqn", "qrdqn", "ppo", "a2c", "maskable-ppo"],
        )

    def test_get_algorithm_defaults_when_none(self):
        self.assertEqual(rl_algorithms.get_algorithm(None).key, "dqn")

    def test_get_algorithm_rejects_unknown(self):
        with self.assertRaisesRegex(ValueError, "Unknown algorithm"):
            rl_algorithms.get_algorithm("does-not-exist")

    def test_every_spec_is_browser_exportable_mlp(self):
        for key in rl_algorithms.algorithm_keys():
            spec = rl_algorithms.get_algorithm(key)
            self.assertEqual(spec.key, key)
            self.assertTrue(spec.export_modules, f"{key} must declare export modules")
            self.assertIn(spec.output_head, {"q-values", "logits", "quantiles"})
            self.assertIn(spec.browser_runtime, {"js-mlp", "ort-web"})
            self.assertTrue(spec.manifest_algorithm)

    def test_dqn_export_path_is_unchanged(self):
        spec = rl_algorithms.get_algorithm("dqn")
        self.assertEqual(spec.export_modules, ("q_net",))
        self.assertEqual(spec.output_head, "q-values")
        self.assertFalse(spec.action_masking)

    def test_only_maskable_ppo_masks_actions(self):
        masking = {k for k in rl_algorithms.algorithm_keys() if rl_algorithms.get_algorithm(k).action_masking}
        self.assertEqual(masking, {"maskable-ppo"})

    def test_actor_critic_methods_export_extractor_plus_head(self):
        for key in ("ppo", "a2c", "maskable-ppo"):
            spec = rl_algorithms.get_algorithm(key)
            self.assertEqual(spec.export_modules, ("mlp_extractor.policy_net", "action_net"))
            self.assertFalse(spec.off_policy)

    def test_checkpoint_dir_name_namespaces_non_dqn(self):
        self.assertEqual(rl_algorithms.checkpoint_dir_name("dqn", 15), "galagai-balanced-v15")
        self.assertEqual(rl_algorithms.checkpoint_dir_name("ppo", 15), "galagai-balanced-v15-ppo")


if __name__ == "__main__":
    unittest.main()
