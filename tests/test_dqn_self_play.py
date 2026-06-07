import unittest
import tempfile
from pathlib import Path

import numpy as np

from alien_invasion.DQN import (
    DQNAgent,
    DQNConfig,
    FRAME_SHAPE,
    ReplayBuffer,
    SelfPlaySchedule,
    TrainingSnapshot,
    Transition,
    enemy_action_to_move,
    enemy_reward,
    normalize_state,
    one_hot,
    pilot_reward,
)


class DqnSelfPlayTest(unittest.TestCase):
    def test_schedule_alternates_roles_by_episode(self):
        schedule = SelfPlaySchedule(alternate_every=2)

        roles = [schedule.role_for_episode(episode) for episode in range(1, 7)]

        self.assertEqual(roles, ["pilot", "pilot", "enemy", "enemy", "pilot", "pilot"])

    def test_replay_buffer_caps_capacity_and_samples(self):
        buffer = ReplayBuffer(capacity=2)
        state = np.zeros(FRAME_SHAPE, dtype=np.float32)

        for action in range(3):
            buffer.append(Transition(state, action, float(action), state, False))

        self.assertEqual(len(buffer), 2)
        sample = buffer.sample(5)
        self.assertEqual(len(sample), 2)
        self.assertTrue(all(item.action in {1, 2} for item in sample))

    def test_normalize_state_accepts_legacy_flat_frame(self):
        flat = np.ones(68 * 52, dtype=np.uint8)

        normalized = normalize_state(flat)

        self.assertEqual(normalized.shape, FRAME_SHAPE)
        self.assertEqual(normalized.dtype, np.float32)

    def test_action_helpers_validate_discrete_spaces(self):
        self.assertTrue(np.array_equal(one_hot(2, 4), np.array([0, 0, 1, 0], dtype=np.float32)))
        self.assertEqual(enemy_action_to_move(0), [-1, 1])
        self.assertEqual(enemy_action_to_move(1), [0, 1])
        self.assertEqual(enemy_action_to_move(2), [1, 1])
        self.assertEqual(enemy_action_to_move(3), [0, 1])

        with self.assertRaises(ValueError):
            one_hot(4, 4)
        with self.assertRaises(ValueError):
            enemy_action_to_move(4)

    def test_pilot_and_enemy_rewards_are_opposed(self):
        before = TrainingSnapshot(score=100, ships_left=2, aliens_left=8)
        after = TrainingSnapshot(score=150, ships_left=1, aliens_left=7)

        self.assertLess(pilot_reward(before, after, done=False), 0)
        self.assertGreater(enemy_reward(before, after, done=False), 0)

    def test_numpy_agent_trains_and_round_trips_checkpoint(self):
        config = DQNConfig(
            state_shape=(3,),
            action_count=2,
            include_bias=False,
            min_replay_size=2,
            batch_size=2,
            learning_rate=0.05,
        )
        agent = DQNAgent(
            config=config,
            weights_path=Path("/tmp/unused-test-agent.npz"),
            seed=7,
            load_existing=False,
            warn_legacy=False,
        )
        before = agent.weights.copy()
        agent.remember(np.array([1, 0, 0]), 0, 1.0, np.array([0, 1, 0]), False)
        agent.remember(np.array([0, 1, 0]), 1, -1.0, np.array([0, 0, 1]), True)

        loss = agent.train()

        self.assertIsNotNone(loss)
        self.assertFalse(np.array_equal(before, agent.weights))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = agent.save(Path(tmpdir) / "agent.npz")
            loaded = DQNAgent(
                config=config,
                weights_path=path,
                seed=8,
                load_existing=True,
                warn_legacy=False,
            )
        self.assertTrue(np.allclose(agent.weights, loaded.weights))


if __name__ == "__main__":
    unittest.main()
