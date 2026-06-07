"""Modern DQN utilities for Alien Invasion self-play training.

The browser demo uses a tiny static model so GitHub Pages can run without a ML
stack. This module is the heavier local-training path for the Pygame game. It
keeps Keras optional at import time so non-ML tests can still run on machines
without TensorFlow wheels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Deque, Iterable

import numpy as np


FRAME_SHAPE = (68, 52, 1)
FLAT_FRAME_SIZE = FRAME_SHAPE[0] * FRAME_SHAPE[1]
PILOT_ACTIONS = ("right", "left", "fire", "hold")
ENEMY_ACTIONS = ("drift_left", "drop", "drift_right")
DEFAULT_CHECKPOINT_DIR = Path(__file__).with_name("checkpoints")
DEFAULT_PILOT_WEIGHTS = DEFAULT_CHECKPOINT_DIR / "pilot.weights.h5"
DEFAULT_ENEMY_WEIGHTS = DEFAULT_CHECKPOINT_DIR / "enemy.weights.h5"
LEGACY_WEIGHTS = Path(__file__).with_name("weights.hdf5")


@dataclass(frozen=True)
class DQNConfig:
    """Configuration for the dueling Double-DQN implementation."""

    state_shape: tuple[int, int, int] = FRAME_SHAPE
    action_count: int = len(PILOT_ACTIONS)
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 50_000
    replay_capacity: int = 50_000
    min_replay_size: int = 1_000
    batch_size: int = 32
    target_sync_interval: int = 500
    gradient_steps: int = 1


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass(frozen=True)
class TrainingSnapshot:
    score: int
    ships_left: int
    aliens_left: int


@dataclass(frozen=True)
class SelfPlayCheckpoint:
    role: str
    episode: int
    weights_path: Path
    score: int
    epsilon: float


class ReplayBuffer:
    """Fixed-capacity replay memory with a small sampling interface."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive.")
        self._items: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, size: int) -> list[Transition]:
        if size <= 0:
            raise ValueError("ReplayBuffer sample size must be positive.")
        return random.sample(list(self._items), min(size, len(self._items)))


class SelfPlaySchedule:
    """Alternates trainable roles while keeping the opponent frozen."""

    roles = ("pilot", "enemy")

    def __init__(self, alternate_every: int = 1):
        if alternate_every <= 0:
            raise ValueError("alternate_every must be positive.")
        self.alternate_every = alternate_every

    def role_for_episode(self, episode_index: int) -> str:
        if episode_index <= 0:
            raise ValueError("episode_index is 1-based and must be positive.")
        role_index = ((episode_index - 1) // self.alternate_every) % len(self.roles)
        return self.roles[role_index]


def _load_keras():
    try:
        import keras
        from keras import layers, losses, ops, optimizers
    except ImportError as error:  # pragma: no cover - depends on optional ML stack.
        raise ImportError(
            "DQNAgent requires Keras 3 with a TensorFlow backend. Use Python "
            "3.10-3.13 and install the ML environment with "
            "`python -m pip install -r requirements-ml.txt`."
        ) from error
    return keras, layers, losses, ops, optimizers


def normalize_state(state: np.ndarray | Iterable[float]) -> np.ndarray:
    """Return one frame as float32 with shape ``(68, 52, 1)``."""

    array = np.asarray(state, dtype=np.float32)
    if array.shape == FRAME_SHAPE:
        return array
    if array.shape == FRAME_SHAPE[:2]:
        return array[..., np.newaxis]
    if array.shape == (FLAT_FRAME_SIZE,):
        return array.reshape(FRAME_SHAPE)
    raise ValueError(f"Expected state shape {FRAME_SHAPE}, {(68, 52)}, or {(FLAT_FRAME_SIZE,)}, got {array.shape}.")


def batch_states(states: Iterable[np.ndarray]) -> np.ndarray:
    return np.stack([normalize_state(state) for state in states], axis=0)


def one_hot(action: int, action_count: int) -> np.ndarray:
    if action < 0 or action >= action_count:
        raise ValueError(f"Action {action} is outside 0..{action_count - 1}.")
    encoded = np.zeros(action_count, dtype=np.float32)
    encoded[action] = 1.0
    return encoded


def action_from_vector(action: int | Iterable[float]) -> int:
    if isinstance(action, (int, np.integer)):
        return int(action)
    return int(np.argmax(np.asarray(action)))


def enemy_action_to_move(action: int) -> list[int]:
    """Map the enemy agent's discrete action to the fleet move interface."""

    moves = {
        0: [-1, 1],
        1: [0, 1],
        2: [1, 1],
    }
    if action not in moves:
        raise ValueError(f"Unknown enemy action {action}.")
    return moves[action]


def pilot_reward(before: TrainingSnapshot, after: TrainingSnapshot, done: bool) -> float:
    score_gain = (after.score - before.score) / 50.0
    life_loss = before.ships_left - after.ships_left
    alien_progress = before.aliens_left - after.aliens_left
    reward = score_gain + 0.05 * alien_progress - 5.0 * max(0, life_loss) - 0.01
    if done and after.ships_left <= 0:
        reward -= 5.0
    return float(reward)


def enemy_reward(before: TrainingSnapshot, after: TrainingSnapshot, done: bool) -> float:
    score_gain = (after.score - before.score) / 50.0
    life_loss = before.ships_left - after.ships_left
    alien_loss = before.aliens_left - after.aliens_left
    reward = -score_gain - 0.15 * max(0, alien_loss) + 5.0 * max(0, life_loss) + 0.01
    if done and after.ships_left <= 0:
        reward += 5.0
    return float(reward)


class DQNAgent:
    """Dueling Double-DQN agent with replay, target network, and checkpoints."""

    def __init__(
        self,
        *,
        config: DQNConfig | None = None,
        weights_path: Path | str | None = None,
        name: str = "pilot",
        seed: int | None = None,
    ):
        self.config = config or DQNConfig()
        self.name = name
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_PILOT_WEIGHTS
        self.random = random.Random(seed)
        self.replay = ReplayBuffer(self.config.replay_capacity)
        self.training_steps = 0
        self.episodes = 0
        self.last_loss: float | None = None

        self.keras, self.layers, self.losses, self.ops, self.optimizers = _load_keras()
        self.model = self._build_network()
        self.target_model = self._build_network()
        self.target_model.set_weights(self.model.get_weights())

        if self.weights_path.exists():
            self.model.load_weights(str(self.weights_path))
            self.target_model.set_weights(self.model.get_weights())
        elif LEGACY_WEIGHTS.exists() and self.name == "pilot":
            print(
                f"Legacy weights found at {LEGACY_WEIGHTS}; new DQN checkpoints "
                f"use {self.weights_path}."
            )

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.training_steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + progress * (self.config.epsilon_final - self.config.epsilon_start)

    def _build_network(self):
        inputs = self.keras.Input(shape=self.config.state_shape, name=f"{self.name}_frame")
        x = self.layers.Rescaling(1.0, name="binary_frame")(inputs)
        x = self.layers.Conv2D(16, 5, strides=2, activation="relu", padding="same", name="conv_1")(x)
        x = self.layers.Conv2D(32, 3, strides=2, activation="relu", padding="same", name="conv_2")(x)
        x = self.layers.Flatten(name="flatten")(x)
        x = self.layers.Dense(128, activation="relu", name="features")(x)

        value = self.layers.Dense(64, activation="relu", name="value_hidden")(x)
        value = self.layers.Dense(1, name="state_value")(value)
        advantage = self.layers.Dense(64, activation="relu", name="advantage_hidden")(x)
        advantage = self.layers.Dense(self.config.action_count, name="advantage")(advantage)
        centered = self.layers.Lambda(
            lambda tensor: tensor - self.ops.mean(tensor, axis=1, keepdims=True),
            name="center_advantage",
        )(advantage)
        q_values = self.layers.Add(name="q_values")([value, centered])

        model = self.keras.Model(inputs=inputs, outputs=q_values, name=f"{self.name}_dueling_dqn")
        model.compile(
            optimizer=self.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=self.losses.Huber(),
        )
        return model

    def act(self, state: np.ndarray, *, training: bool = True) -> int:
        if training and self.random.random() < self.epsilon:
            return self.random.randrange(self.config.action_count)
        q_values = self.model.predict(normalize_state(state)[np.newaxis, ...], verbose=0)[0]
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done) -> None:
        self.replay.append(
            Transition(
                normalize_state(state),
                action_from_vector(action),
                float(reward),
                normalize_state(next_state),
                bool(done),
            )
        )

    def train_short_memory(self, state, action, reward, next_state, done) -> float | None:
        self.remember(state, action, reward, next_state, done)
        return self.train()

    def replay_new(self, memory=None) -> float | None:
        return self.train(gradient_steps=self.config.gradient_steps)

    def train(self, *, gradient_steps: int | None = None) -> float | None:
        if len(self.replay) < self.config.min_replay_size:
            return None

        losses: list[float] = []
        steps = gradient_steps or self.config.gradient_steps
        for _ in range(steps):
            batch = self.replay.sample(self.config.batch_size)
            states = batch_states(item.state for item in batch)
            next_states = batch_states(item.next_state for item in batch)
            actions = np.asarray([item.action for item in batch], dtype=np.int64)
            rewards = np.asarray([item.reward for item in batch], dtype=np.float32)
            dones = np.asarray([item.done for item in batch], dtype=np.float32)

            targets = self.model.predict(states, verbose=0)
            next_online = self.model.predict(next_states, verbose=0)
            next_actions = np.argmax(next_online, axis=1)
            next_target = self.target_model.predict(next_states, verbose=0)
            next_values = next_target[np.arange(len(batch)), next_actions]
            targets[np.arange(len(batch)), actions] = rewards + (1.0 - dones) * self.config.gamma * next_values

            history = self.model.fit(states, targets, epochs=1, verbose=0)
            losses.append(float(history.history["loss"][-1]))
            self.training_steps += 1
            if self.training_steps % self.config.target_sync_interval == 0:
                self.sync_target_network()

        self.last_loss = float(np.mean(losses))
        return self.last_loss

    def sync_target_network(self) -> None:
        self.target_model.set_weights(self.model.get_weights())

    def save(self, path: Path | str | None = None) -> Path:
        destination = Path(path) if path else self.weights_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(destination))
        return destination

    def set_reward(self, score, beforescore, ships_left):
        before = TrainingSnapshot(score=int(beforescore), ships_left=3, aliens_left=0)
        after = TrainingSnapshot(score=int(score), ships_left=int(ships_left), aliens_left=0)
        return pilot_reward(before, after, done=False)


class AlternatingSelfPlayTrainer:
    """Coordinates pilot and enemy agents without leaking role logic to callers."""

    def __init__(
        self,
        *,
        checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
        pilot_weights_path: Path | str | None = None,
        enemy_weights_path: Path | str | None = None,
        alternate_every: int = 1,
        seed: int | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.schedule = SelfPlaySchedule(alternate_every=alternate_every)
        base_seed = seed or 0
        self.pilot = DQNAgent(
            config=DQNConfig(action_count=len(PILOT_ACTIONS)),
            weights_path=pilot_weights_path or self.checkpoint_dir / "pilot.weights.h5",
            name="pilot",
            seed=base_seed,
        )
        self.enemy = DQNAgent(
            config=DQNConfig(action_count=len(ENEMY_ACTIONS)),
            weights_path=enemy_weights_path or self.checkpoint_dir / "enemy.weights.h5",
            name="enemy",
            seed=base_seed + 1,
        )

    def role_for_episode(self, episode_index: int) -> str:
        return self.schedule.role_for_episode(episode_index)

    def agent_for_role(self, role: str) -> DQNAgent:
        if role == "pilot":
            return self.pilot
        if role == "enemy":
            return self.enemy
        raise ValueError(f"Unknown self-play role {role!r}.")

    def opponent_for_role(self, role: str) -> DQNAgent:
        return self.enemy if role == "pilot" else self.pilot

    def checkpoint(self, *, role: str, episode: int, score: int) -> SelfPlayCheckpoint:
        agent = self.agent_for_role(role)
        path = agent.save(self.checkpoint_dir / f"{role}-episode-{episode:05d}.weights.h5")
        agent.save()
        return SelfPlayCheckpoint(role=role, episode=episode, weights_path=path, score=score, epsilon=agent.epsilon)
