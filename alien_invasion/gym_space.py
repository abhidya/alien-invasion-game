"""Gymnasium reference for future Alien Invasion environment adapters.

This file used to execute an unrelated Atari CEM example at import time with
old ``gym`` and ``keras-rl`` imports. It is now a non-executing reference for
the adapter shape needed before wrapping the game as a modern Gymnasium env.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GymnasiumAdapterSpec:
    """Interface facts a Gymnasium adapter must satisfy."""

    observation_shape: tuple[int, ...] = (11,)
    pilot_actions: tuple[str, ...] = ("left", "right", "fire", "stay")
    enemy_actions: tuple[str, ...] = ("drift_left", "drop", "drift_right", "fire")
    reset_returns_info: bool = True
    step_returns_terminated_truncated: bool = True
    render_mode_required_at_make: bool = True


GYMNASIUM_REFERENCE = {
    "environment_import": "import gymnasium as gym",
    "local_trainer": "tools.train_static_pilot.train_self_play",
    "why": (
        "Alien Invasion has discrete pilot and enemy actions. The static "
        "trainer now uses a Gymnasium-compatible headless game loop and "
        "alternating Stable-Baselines3 DQN agents."
    ),
    "notes": [
        "Use a separate evaluation environment, not the training loop.",
        "Handle terminated and truncated separately in custom adapters.",
        "Freeze the opponent policy while training the active self-play role.",
        "Rate-limit and penalize invalid enemy drop actions to avoid drop spam.",
    ],
}


def describe_adapter() -> dict[str, object]:
    spec = GymnasiumAdapterSpec()
    return {
        "observation_shape": spec.observation_shape,
        "pilot_action_count": len(spec.pilot_actions),
        "enemy_action_count": len(spec.enemy_actions),
        "gymnasium_reset": "obs, info = env.reset(seed=seed)",
        "gymnasium_step": "obs, reward, terminated, truncated, info = env.step(action)",
        "gymnasium": GYMNASIUM_REFERENCE,
    }
