import json
import tempfile
import unittest
from pathlib import Path

from tools import train_static_pilot


class StaticPilotArtifactTest(unittest.TestCase):
    def test_environment_penalizes_drop_spam(self):
        env = train_static_pilot.HeadlessGalagai(seed=12, max_steps=20)

        _, _, first_enemy_reward, _, first_info = env.step(3, 2)
        _, _, second_enemy_reward, _, second_info = env.step(3, 2)

        self.assertTrue(first_info["events"].valid_drop)
        self.assertFalse(first_info["events"].invalid_drop)
        self.assertTrue(second_info["events"].invalid_drop)
        self.assertLess(second_enemy_reward, first_enemy_reward)
        self.assertEqual(second_info["invalidDrops"], 1)

    def test_wave_one_bees_cannot_fire(self):
        env = train_static_pilot.HeadlessGalagai(seed=14, max_steps=20)

        _, _, _, _, info = env.step(3, 5)

        self.assertTrue(info["events"].invalid_fire)
        self.assertFalse(info["events"].enemy_fired)
        self.assertEqual(info["enemyFires"], 0)

    def test_observation_exposes_incoming_shots_and_pilot_bullets(self):
        env = train_static_pilot.HeadlessGalagai(seed=16, max_steps=20)
        ship_center = env.ship.center_x
        env.enemy_shots = [
            train_static_pilot.Shot(ship_center + 220, 520, 6, 16, train_static_pilot.ENEMY_SHOT_SPEED),
            train_static_pilot.Shot(ship_center - 3, 250, 6, 16, train_static_pilot.ENEMY_SHOT_SPEED),
        ]

        features = env.features()

        self.assertAlmostEqual(float(features[11]), 0.0, delta=0.02)
        self.assertGreater(float(features[12]), 0.4)
        self.assertGreater(float(features[13]), 0.9)

        threatened_alien = next(alien for alien in env.aliens if alien.alive)
        env.bullets = [
            train_static_pilot.Shot(
                threatened_alien.center_x - 3,
                threatened_alien.bottom + 70,
                6,
                18,
                train_static_pilot.BULLET_SPEED,
            )
        ]

        features = env.features()

        self.assertAlmostEqual(float(features[14]), 0.0, delta=0.02)
        self.assertGreater(float(features[15]), 0.0)
        self.assertGreater(float(features[16]), 0.0)

    def test_opening_enemy_policy_respects_wave_roles(self):
        policy = train_static_pilot.OpeningEnemyPolicy()
        wave_one = train_static_pilot.HeadlessGalagai(seed=17, max_steps=20)

        self.assertNotIn(policy.act(wave_one.features()), {1, 4, 5, 6, 7})

        wave_two = train_static_pilot.HeadlessGalagai(seed=18, max_steps=20)
        wave_two.start_wave = 2
        wave_two.wave = 2
        wave_two.aliens = wave_two.create_fleet(2)

        action = policy.act(wave_two.features())
        _, _, _, _, info = wave_two.step(3, action)

        self.assertIn(action, {1, 4, 5})
        self.assertTrue(info["events"].enemy_fired)

    def test_curriculum_start_wave_does_not_count_as_free_pilot_win(self):
        env = train_static_pilot.HeadlessGalagai(seed=19, max_steps=20, max_start_wave=3)
        env.start_wave = 3
        env.wave = 3
        env.steps = env.max_steps

        self.assertEqual(env.winner(done=True), "timeout")

    def test_partial_score_does_not_count_as_pilot_dominance(self):
        env = train_static_pilot.HeadlessGalagai(seed=15, max_steps=20)
        env.score = 700
        env.steps = env.max_steps

        self.assertEqual(env.winner(done=True), "timeout")

    def test_pilot_reward_prefers_hits_and_fast_wave_clears(self):
        env = train_static_pilot.HeadlessGalagai(seed=13, max_steps=100)
        accurate_hit = train_static_pilot.StepEvents(
            score_gain=50,
            aliens_destroyed=1,
            life_loss=0,
            wave_cleared=False,
            invalid_drop=False,
            invalid_fire=False,
            valid_drop=False,
            enemy_fired=False,
            pilot_fired=True,
            pilot_missed=False,
            pilot_hits=1,
            pilot_aligned_action=True,
            pilot_aligned_fire=True,
            pilot_bad_fire=False,
            enemy_tactical_action=False,
        )
        off_target_fire = train_static_pilot.StepEvents(
            score_gain=0,
            aliens_destroyed=0,
            life_loss=0,
            wave_cleared=False,
            invalid_drop=False,
            invalid_fire=False,
            valid_drop=False,
            enemy_fired=False,
            pilot_fired=True,
            pilot_missed=False,
            pilot_hits=0,
            pilot_aligned_action=False,
            pilot_aligned_fire=False,
            pilot_bad_fire=True,
            enemy_tactical_action=False,
        )
        slow_clear = train_static_pilot.StepEvents(
            score_gain=250,
            aliens_destroyed=0,
            life_loss=0,
            wave_cleared=True,
            invalid_drop=False,
            invalid_fire=False,
            valid_drop=False,
            enemy_fired=False,
            pilot_fired=False,
            pilot_missed=False,
            pilot_hits=0,
            pilot_aligned_action=False,
            pilot_aligned_fire=False,
            pilot_bad_fire=False,
            enemy_tactical_action=False,
        )

        self.assertGreater(env.pilot_reward(accurate_hit, done=False), env.pilot_reward(off_target_fire, done=False))
        env.steps = 10
        fast_clear_reward = env.pilot_reward(slow_clear, done=True)
        env.steps = 90
        slow_clear_reward = env.pilot_reward(slow_clear, done=True)

        self.assertGreater(fast_clear_reward, slow_clear_reward)

    def test_balanced_role_selector_trains_dominated_side(self):
        checkpoints = {"pilot": [{"id": 1}], "enemies": [{"id": 1}]}

        self.assertEqual(
            train_static_pilot.choose_balanced_role(
                {"pilotWinRate": 0.0, "enemyWinRate": 1.0},
                checkpoints,
                dominance_threshold=0.6,
            ),
            "pilot",
        )
        self.assertEqual(
            train_static_pilot.choose_balanced_role(
                {"pilotWinRate": 1.0, "enemyWinRate": 0.0},
                checkpoints,
                dominance_threshold=0.6,
            ),
            "enemies",
        )
        self.assertEqual(
            train_static_pilot.choose_balanced_role(
                {"pilotWinRate": 0.0, "enemyWinRate": 0.0},
                checkpoints,
                dominance_threshold=0.6,
            ),
            "pilot",
        )

    def test_balanced_stop_requires_recent_competitive_balance(self):
        history = [
            {"pilotWinRate": 0.45, "enemyWinRate": 0.35},
            {"pilotWinRate": 0.4, "enemyWinRate": 0.35},
        ]

        self.assertTrue(
            train_static_pilot.balanced_stop_reached(
                history,
                min_balanced_rounds=2,
                balance_patience=2,
                dominance_threshold=0.6,
                balance_tolerance=0.18,
                balance_min_win_rate=0.25,
            )
        )
        self.assertFalse(
            train_static_pilot.balanced_stop_reached(
                [{"pilotWinRate": 0.0, "enemyWinRate": 0.0}, {"pilotWinRate": 0.0, "enemyWinRate": 0.0}],
                min_balanced_rounds=2,
                balance_patience=2,
                dominance_threshold=0.6,
                balance_tolerance=0.18,
                balance_min_win_rate=0.25,
            )
        )

    def test_tiered_retention_keeps_latest_and_sparse_history(self):
        retention = train_static_pilot.CheckpointRetention(mode="tiered", keep_latest=5)

        retained = train_static_pilot.retained_generation_ids(130, retention)

        self.assertIn(1, retained)
        self.assertIn(2, retained)
        self.assertIn(100, retained)
        self.assertIn(110, retained)
        self.assertIn(120, retained)
        self.assertIn(126, retained)
        self.assertIn(130, retained)
        self.assertNotIn(3, retained)
        self.assertNotIn(101, retained)
        self.assertNotIn(111, retained)

    def test_checkpoint_store_prunes_export_files_without_reusing_generation_ids(self):
        retention = train_static_pilot.CheckpointRetention(mode="tiered", keep_latest=2)
        entries = [{"id": index, "role": "pilot"} for index in range(1, 8)]
        history = [{"trained": "pilot", "generation": index} for index in range(1, 8)]

        with tempfile.TemporaryDirectory() as tmpdir:
            store = train_static_pilot.TrainingCheckpointStore(Path(tmpdir), retention)
            store.save(
                pilot_model=None,
                enemy_model=None,
                history=history,
                checkpoints={"pilot": entries, "enemies": []},
                round_number=7,
                phase_number=3,
                config={},
            )
            state = json.loads((Path(tmpdir) / "state.json").read_text(encoding="utf-8"))
            files = sorted(path.name for path in (Path(tmpdir) / "exports").glob("pilot-v*.json"))

        self.assertEqual(files, ["pilot-v001.json", "pilot-v002.json", "pilot-v004.json", "pilot-v006.json", "pilot-v007.json"])
        self.assertEqual(state["checkpointCounts"], {"pilot": 5, "enemies": 0})
        self.assertEqual(state["totalGenerationCounts"], {"pilot": 7, "enemies": 0})
        self.assertEqual(train_static_pilot.role_generation_count(history, "pilot") + 1, 8)

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

        self.assertEqual(payload["version"], 11)
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
        self.assertIn("pilotShotAccuracy", payload["metrics"])
        self.assertIn("waveClearRate", payload["metrics"])

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

    def test_checkpoint_resume_continues_generation_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            train_static_pilot.train_self_play(
                seed=51,
                generations_per_side=1,
                phase_timesteps=32,
                eval_episodes=1,
                max_steps=40,
                dominance_threshold=1.1,
                max_phase_iterations=1,
                checkpoint_dir=checkpoint_dir,
            )

            _, _, resumed = train_static_pilot.train_self_play(
                seed=51,
                generations_per_side=2,
                phase_timesteps=32,
                eval_episodes=1,
                max_steps=40,
                dominance_threshold=1.1,
                max_phase_iterations=1,
                checkpoint_dir=checkpoint_dir,
                resume=True,
            )

            state = json.loads((checkpoint_dir / "state.json").read_text(encoding="utf-8"))

        self.assertTrue(resumed["resumedFromCheckpoint"])
        self.assertEqual(len(resumed["checkpoints"]["pilot"]), 2)
        self.assertEqual(len(resumed["checkpoints"]["enemies"]), 2)
        self.assertEqual(state["checkpointCounts"], {"pilot": 2, "enemies": 2})
        self.assertEqual(len(state["checkpointFiles"]["pilot"]), 2)
        self.assertEqual(len(state["checkpointFiles"]["enemies"]), 2)

    def test_parallel_evaluation_workers_score_policy_specs(self):
        metrics = train_static_pilot.evaluate_policy_specs(
            seed=61,
            pilot_spec={"kind": "heuristic-pilot"},
            enemy_spec={"kind": "opening-enemy"},
            episodes=2,
            max_steps=12,
            workers=2,
        )

        self.assertIn("pilotWinRate", metrics)
        self.assertIn("enemyWinRate", metrics)
        self.assertGreater(metrics["averageSteps"], 0)


if __name__ == "__main__":
    unittest.main()
