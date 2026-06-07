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

    observation_shape: tuple[int, int, int] = (68, 52, 1)
    pilot_actions: tuple[str, ...] = ("right", "left", "fire", "hold")
    enemy_actions: tuple[str, ...] = ("drift_left", "drop", "drift_right")
    reset_returns_info: bool = True
    step_returns_terminated_truncated: bool = True
    render_mode_required_at_make: bool = True


GYMNASIUM_REFERENCE = {
    "environment_import": "import gymnasium as gym",
    "local_trainer": "alien_invasion.DQN.DQNAgent",
    "why": (
        "Alien Invasion has discrete pilot and enemy actions. The current "
        "Python 3.14-compatible path uses NumPy Q agents; a Gymnasium adapter "
        "would let that same loop expose reset/step/render cleanly."
    ),
    "notes": [
        "Use a separate evaluation environment, not the training loop.",
        "Handle terminated and truncated separately in custom adapters.",
        "Freeze the opponent policy while training the active self-play role.",
        "Stable-Baselines3 remains a good option on Python versions where PyTorch resolves.",
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
