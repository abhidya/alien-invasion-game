import json
import tempfile
import unittest
from pathlib import Path

from tools import train_static_pilot


class StaticPilotArtifactTest(unittest.TestCase):
    def test_relative_x_uses_toroidal_shortest_distance(self):
        rel = train_static_pilot.HeadlessGalagai._relative_x
        width = train_static_pilot.CANVAS_WIDTH
        half = width / 2.0
        # Aligned -> zero.
        self.assertAlmostEqual(rel(0.0), 0.0)
        # Small linear deltas are unchanged.
        self.assertAlmostEqual(rel(10.0), 10.0 / half)
        self.assertAlmostEqual(rel(-10.0), -10.0 / half)
        # A delta just under the full width is physically a short hop across the
        # seam: it must wrap to a small magnitude, not saturate near +/-1.
        self.assertAlmostEqual(rel(width - 20.0), -20.0 / half)
        self.assertAlmostEqual(rel(-(width - 20.0)), 20.0 / half)
        # Result always stays within [-1, 1].
        for value in (-2 * width, -width, -half, 0.0, half, width, 2 * width):
            self.assertGreaterEqual(rel(value), -1.0)
            self.assertLessEqual(rel(value), 1.0)

    def test_exported_manifest_marks_grid_feature_encoding(self):
        self.assertEqual(train_static_pilot.FEATURE_ENCODING, "grid-v1")
        self.assertEqual(train_static_pilot.MODEL_SCHEMA_VERSION, 16)
        self.assertEqual(train_static_pilot.FRAME_SHAPE, (8, 28, 48))
        self.assertEqual(len(train_static_pilot.FEATURES), 8 * 28 * 48 + 6)

    def test_grid_observation_marks_entities_and_scalars(self):
        env = train_static_pilot.HeadlessGalagai(seed=101, max_steps=20)
        alien = env.aliens[0]
        env.bullets = [
            train_static_pilot.Shot(
                env.ship.center_x - 3,
                env.ship.y - 40,
                6,
                18,
                train_static_pilot.BULLET_SPEED,
            )
        ]
        env.enemy_shots = [
            train_static_pilot.Shot(
                env.ship.center_x - 3,
                env.ship.y - 100,
                6,
                16,
                train_static_pilot.ENEMY_SHOT_SPEED,
            )
        ]

        observation = env.features(alien)
        frame_size = int(__import__("numpy").prod(train_static_pilot.FRAME_SHAPE))
        frame = observation[:frame_size].reshape(train_static_pilot.FRAME_SHAPE)
        scalars = observation[frame_size:]

        self.assertEqual(observation.shape, (len(train_static_pilot.FEATURES),))
        self.assertGreater(frame[0].sum(), 0.0)
        self.assertGreater(frame[1].sum(), 0.0)
        self.assertGreater(frame[2].sum(), 0.0)
        self.assertGreater(frame[3].sum() + frame[4].sum() + frame[5].sum(), 0.0)
        self.assertGreater(frame[6].sum(), 0.0)
        self.assertGreater(frame[7].sum(), 0.0)
        self.assertEqual(len(scalars), len(train_static_pilot.SCALAR_FEATURES))

    def test_environment_penalizes_drop_spam(self):
        env = train_static_pilot.HeadlessGalagai(seed=12, max_steps=20)
        alien = env.aliens[0]

        _, _, first_enemy_reward, _, first_info = env.step(3, [(alien, 3)])
        _, _, second_enemy_reward, _, second_info = env.step(3, [(alien, 3)])

        self.assertTrue(first_info["events"].valid_drop)
        self.assertFalse(first_info["events"].invalid_drop)
        self.assertTrue(second_info["events"].invalid_drop)
        self.assertLess(second_enemy_reward, first_enemy_reward)
        self.assertEqual(second_info["invalidDrops"], 1)

    def test_bees_cannot_fire_but_shooters_are_armed_from_wave_one(self):
        # No first-wave nerf: shooter roles exist on wave 1 and can fire. Firing
        # is gated by role (bee vs butterfly/boss), never by the wave number.
        env = train_static_pilot.HeadlessGalagai(seed=14, max_steps=20)

        bee = next(alien for alien in env.aliens if alien.role == "bee")
        shooter = next(alien for alien in env.aliens if alien.role in ("butterfly", "boss"))

        bee_events = env.apply_enemy_actions([(bee, 7)])
        self.assertEqual(bee_events["enemy_fires"], 0)
        self.assertEqual(bee_events["invalid_fires"], 1)

        shooter_events = env.apply_enemy_actions([(shooter, 7)])
        self.assertEqual(shooter_events["enemy_fires"], 1)
        self.assertEqual(len(env.enemy_shots), 1)

    def test_enemy_reaching_bottom_dies_without_pilot_life_loss(self):
        env = train_static_pilot.HeadlessGalagai(seed=20, max_steps=20)
        alien = env.aliens[0]
        alien.x = 18
        alien.y = train_static_pilot.CANVAS_HEIGHT - alien.height + 1
        lives_before = env.lives

        env.update_aliens()

        self.assertFalse(alien.alive)
        self.assertEqual(env.lives, lives_before)

    def test_enemy_wraps_horizontally_instead_of_resetting(self):
        env = train_static_pilot.HeadlessGalagai(seed=22, max_steps=20)
        alien = env.aliens[0]
        alien.x = train_static_pilot.CANVAS_WIDTH + 1

        env.wrap_alien_horizontal(alien)

        self.assertEqual(alien.x, -alien.width)

        alien.x = -alien.width - 1
        env.wrap_alien_horizontal(alien)

        self.assertEqual(alien.x, train_static_pilot.CANVAS_WIDTH)

    def test_enemy_ai_controls_individual_ships(self):
        env = train_static_pilot.HeadlessGalagai(seed=26, max_steps=20)
        first = env.aliens[0]
        second = env.aliens[1]
        first_x = first.x
        second_x = second.x

        env.apply_enemy_actions([(first, 1), (second, 2)])

        self.assertLess(first.x, first_x)
        self.assertGreater(second.x, second_x)

    def test_multiple_enemy_ships_can_fire_independently(self):
        env = train_static_pilot.HeadlessGalagai(seed=27, max_steps=20)
        env.wave = 2
        env.aliens = env.create_fleet(2)
        shooters = [alien for alien in env.aliens if alien.role == "butterfly"][:2]

        events = env.apply_enemy_actions([(shooters[0], 7), (shooters[1], 7)])

        self.assertEqual(events["enemy_fires"], 2)
        self.assertEqual(len(env.enemy_shots), 2)

    def test_pilot_wraps_horizontally_instead_of_clamping(self):
        env = train_static_pilot.HeadlessGalagai(seed=25, max_steps=20)

        env.ship.x = train_static_pilot.CANVAS_WIDTH + 1
        env.wrap_ship_horizontal()

        self.assertEqual(env.ship.x, -env.ship.width)

        env.ship.x = -env.ship.width - 1
        env.wrap_ship_horizontal()

        self.assertEqual(env.ship.x, train_static_pilot.CANVAS_WIDTH)

    def test_enemy_crashing_into_pilot_kills_both(self):
        env = train_static_pilot.HeadlessGalagai(seed=23, max_steps=20)
        alien = env.aliens[0]
        alien.x = env.ship.x
        alien.y = env.ship.y
        lives_before = env.lives

        env.update_aliens()

        self.assertFalse(alien.alive)
        self.assertEqual(env.lives, lives_before - 1)

    def test_pilot_can_move_up_and_down_with_bounds(self):
        env = train_static_pilot.HeadlessGalagai(seed=24, max_steps=20)
        start_y = env.ship.y

        env.step(4, 0)
        up_y = env.ship.y
        env.step(5, 0)

        self.assertLess(up_y, start_y)
        self.assertGreater(env.ship.y, up_y)

        env.ship.y = train_static_pilot.SHIP_MIN_Y - 100
        env.step(4, 0)
        self.assertEqual(env.ship.y, train_static_pilot.SHIP_MIN_Y)

        env.ship.y = train_static_pilot.SHIP_MAX_Y + 100
        env.step(5, 0)
        self.assertEqual(env.ship.y, train_static_pilot.SHIP_MAX_Y)

    def test_opening_enemy_policy_is_armed_from_wave_one(self):
        # No first-wave nerf: the controlled enemy is a shooter on wave 1, so the
        # opening policy can fire immediately instead of only after wave one.
        policy = train_static_pilot.OpeningEnemyPolicy()
        env = train_static_pilot.HeadlessGalagai(seed=17, max_steps=20)

        self.assertTrue(any(alien.role in ("butterfly", "boss") for alien in env.aliens))
        self.assertTrue(env.can_enemy_fire(env.selected_enemy()))

        action = policy.act(env.enemy_control_observation())
        _, _, _, _, info = env.step(3, action)

        self.assertIn(action, {4, 5, 7})
        self.assertTrue(info["events"].enemy_fired)

    def test_kill_pressure_pushes_fleet_down_and_ramps(self):
        # Kill pressure shoves the whole live fleet toward the floor on a ramping
        # cadence (interval shrinks, step grows) and resets each wave.
        env = train_static_pilot.HeadlessGalagai(seed=31, max_steps=20)
        ys_before = [alien.y for alien in env.aliens]

        # One tick that does not exhaust the interval leaves the fleet in place.
        env.pressure_cooldown = 2.0 * train_static_pilot.ACTION_DT
        env.apply_kill_pressure()
        self.assertEqual([alien.y for alien in env.aliens], ys_before)

        # Force the interval to elapse on the next tick, then it pushes.
        env.pressure_cooldown = train_static_pilot.ACTION_DT
        first_step = env.pressure_step
        env.apply_kill_pressure()
        self.assertTrue(all(after > before for after, before in zip([a.y for a in env.aliens], ys_before)))
        self.assertAlmostEqual(env.aliens[0].y - ys_before[0], first_step)
        # Ramp: the next interval is shorter and the next step is larger.
        self.assertLess(env.pressure_interval, train_static_pilot.PRESSURE_BASE_INTERVAL)
        self.assertGreater(env.pressure_step, first_step)

        env.reset_kill_pressure()
        self.assertEqual(env.pressure_interval, train_static_pilot.PRESSURE_BASE_INTERVAL)
        self.assertEqual(env.pressure_step, train_static_pilot.PRESSURE_STEP)

    def test_no_wave_progression_beyond_trained_generation(self):
        # Fleet speed and size, and enemy shot speed, are constant across waves;
        # only the trained enemy generation changes difficulty.
        env = train_static_pilot.HeadlessGalagai(seed=32, max_steps=20)
        wave_one = env.create_fleet(1)
        wave_five = env.create_fleet(5)
        self.assertEqual(len(wave_one), len(wave_five))
        self.assertEqual(train_static_pilot.FLEET_SPEED_PER_WAVE, 0.0)
        self.assertEqual(train_static_pilot.ENEMY_SHOT_SPEED_PER_WAVE, 0.0)

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

    def test_balanced_stop_waits_for_required_new_rounds_after_resume(self):
        history = [
            {"pilotWinRate": 0.45, "enemyWinRate": 0.35},
            {"pilotWinRate": 0.4, "enemyWinRate": 0.35},
            {"pilotWinRate": 0.5, "enemyWinRate": 0.4},
        ]

        self.assertFalse(
            train_static_pilot.balanced_stop_reached_after_required_rounds(
                history,
                completed_generations=3,
                required_new_balanced_rounds=1,
                min_balanced_rounds=2,
                balance_patience=2,
                dominance_threshold=0.6,
                balance_tolerance=0.18,
                balance_min_win_rate=0.25,
            )
        )
        self.assertTrue(
            train_static_pilot.balanced_stop_reached_after_required_rounds(
                [*history, {"pilotWinRate": 0.45, "enemyWinRate": 0.35}],
                completed_generations=3,
                required_new_balanced_rounds=1,
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

    def test_candidate_selection_scores_role_specific_metrics(self):
        pilot_results = [
            train_static_pilot.CandidateResult(
                spawn_index=1,
                metrics={"pilotWinRate": 0.1, "waveClearRate": 0.8, "pilotShotAccuracy": 0.9},
                model_path="pilot-1.zip",
                replay_path="pilot-1.pkl",
            ),
            train_static_pilot.CandidateResult(
                spawn_index=2,
                metrics={"pilotWinRate": 0.2, "waveClearRate": 0.1, "pilotShotAccuracy": 0.1},
                model_path="pilot-2.zip",
                replay_path="pilot-2.pkl",
            ),
        ]
        enemy_results = [
            train_static_pilot.CandidateResult(
                spawn_index=1,
                metrics={"enemyWinRate": 0.5, "enemyFireRate": 0.8, "invalidDropRate": 0.0},
                model_path="enemy-1.zip",
                replay_path="enemy-1.pkl",
            ),
            train_static_pilot.CandidateResult(
                spawn_index=2,
                metrics={"enemyWinRate": 0.5, "enemyFireRate": 0.4, "invalidDropRate": 0.0},
                model_path="enemy-2.zip",
                replay_path="enemy-2.pkl",
            ),
        ]

        self.assertEqual(train_static_pilot.best_candidate_result(pilot_results, "pilot").spawn_index, 2)
        self.assertEqual(train_static_pilot.best_candidate_result(enemy_results, "enemies").spawn_index, 1)

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

        self.assertEqual(payload["version"], 16)
        self.assertEqual(payload["featureEncoding"], "grid-v1")
        self.assertEqual(payload["frameShape"], list(train_static_pilot.FRAME_SHAPE))
        self.assertEqual(payload["gridChannels"], train_static_pilot.GRID_CHANNELS)
        self.assertEqual(payload["scalarFeatures"], train_static_pilot.SCALAR_FEATURES)
        self.assertEqual(payload["enemies"]["featureEncoding"], "grid-v1")
        self.assertEqual(payload["enemies"]["frameShape"], list(train_static_pilot.FRAME_SHAPE))
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

    def test_warmup_generations_run_before_balanced_selector(self):
        _, _, self_play = train_static_pilot.train_self_play(
            seed=45,
            balanced_rounds=4,
            pilot_warmup_generations=3,
            enemy_warmup_generations=1,
            phase_timesteps=32,
            eval_episodes=1,
            max_steps=40,
            dominance_threshold=1.1,
            max_phase_iterations=1,
        )

        self.assertEqual(
            [round_info["trained"] for round_info in self_play["rounds"]],
            ["pilot", "pilot", "pilot", "enemies"],
        )
        self.assertEqual(len(self_play["checkpoints"]["pilot"]), 3)
        self.assertEqual(len(self_play["checkpoints"]["enemies"]), 1)
        self.assertEqual(self_play["pilotWarmupGenerations"], 3)
        self.assertEqual(self_play["enemyWarmupGenerations"], 1)

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
