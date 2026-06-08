"""Tests for recommendation C: reward shaping as data.

The reward used to be magic numbers welded into the env's step loop and could
only be exercised by running a whole episode. It is now a pure function over a
StepEvents fixture, a RewardContext, and a RewardProfile -- so it is testable in
isolation, and the weights are data you can swap without retraining.
"""

import unittest

from tools.train_static_pilot import (
    DEFAULT_REWARD_PROFILE,
    RewardContext,
    RewardProfile,
    StepEvents,
    compute_enemy_reward,
    compute_pilot_reward,
)


def make_events(**overrides) -> StepEvents:
    base = dict(
        score_gain=0,
        aliens_destroyed=0,
        life_loss=0,
        wave_cleared=False,
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
    base.update(overrides)
    return StepEvents(**base)


CTX = RewardContext(clear_speed=0.0, winner="", live_alien_fraction=0.0)


class RewardProfileTest(unittest.TestCase):
    def test_pilot_idle_step_pays_only_the_step_cost(self):
        r = compute_pilot_reward(make_events(), done=False, ctx=CTX, profile=DEFAULT_REWARD_PROFILE)
        self.assertAlmostEqual(r, -DEFAULT_REWARD_PROFILE.pilot_step_cost)

    def test_pilot_hit_and_life_loss_use_profile_weights(self):
        p = DEFAULT_REWARD_PROFILE
        ev = make_events(score_gain=70, pilot_hits=2, life_loss=1)
        expected = 70 / p.pilot_score_divisor + 2 * p.pilot_hit - p.pilot_life_loss - p.pilot_step_cost
        r = compute_pilot_reward(ev, done=False, ctx=CTX, profile=p)
        self.assertAlmostEqual(r, expected)

    def test_pilot_win_bonus_scales_with_clear_speed(self):
        p = DEFAULT_REWARD_PROFILE
        ctx = RewardContext(clear_speed=0.5, winner="pilot", live_alien_fraction=0.0)
        r = compute_pilot_reward(make_events(), done=True, ctx=ctx, profile=p)
        # -step_cost + win_base + win_speed*0.5
        self.assertAlmostEqual(r, -p.pilot_step_cost + p.pilot_win_base + p.pilot_win_speed * 0.5)

    def test_enemy_reward_rewards_taking_a_life(self):
        p = DEFAULT_REWARD_PROFILE
        r = compute_enemy_reward(make_events(life_loss=1), done=False, ctx=CTX, profile=p)
        self.assertAlmostEqual(r, p.enemy_life_loss)

    def test_profile_is_data_changing_a_weight_changes_the_reward(self):
        ev = make_events(pilot_hits=1)
        base = compute_pilot_reward(ev, done=False, ctx=CTX, profile=DEFAULT_REWARD_PROFILE)
        louder = RewardProfile(pilot_hit=DEFAULT_REWARD_PROFILE.pilot_hit * 2)
        boosted = compute_pilot_reward(ev, done=False, ctx=CTX, profile=louder)
        self.assertGreater(boosted, base)
        self.assertAlmostEqual(boosted - base, DEFAULT_REWARD_PROFILE.pilot_hit)

    def test_default_profile_reproduces_original_constants(self):
        # Guards the published checkpoints: defaults must equal the pre-refactor
        # magic numbers, or old and new agents are no longer comparable.
        p = DEFAULT_REWARD_PROFILE
        self.assertEqual(p.pilot_score_divisor, 35.0)
        self.assertEqual(p.pilot_life_loss, 7.0)
        self.assertEqual(p.pilot_step_cost, 0.008)
        self.assertEqual(p.enemy_score_divisor, 55.0)
        self.assertEqual(p.enemy_alien_destroyed, 1.45)


if __name__ == "__main__":
    unittest.main()
