"""NumPy DQN utilities for Alien Invasion self-play training.

The local workspace is currently Python 3.14, where TensorFlow and PyTorch
wheels are not available through pip. This module keeps the DQN path runnable by
using a linear Q model trained with replay memory, a target network, Huber loss,
and alternating pilot/enemy self-play.
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
ENEMY_ACTIONS = ("drift_left", "drop", "drift_right", "fire")
DEFAULT_CHECKPOINT_DIR = Path(__file__).with_name("checkpoints")
DEFAULT_PILOT_WEIGHTS = DEFAULT_CHECKPOINT_DIR / "pilot.npz"
DEFAULT_ENEMY_WEIGHTS = DEFAULT_CHECKPOINT_DIR / "enemy.npz"
LEGACY_WEIGHTS = Path(__file__).with_name("weights.hdf5")


@dataclass(frozen=True)
class DQNConfig:
    """Configuration for the NumPy linear Double-DQN implementation."""

    state_shape: tuple[int, ...] = FRAME_SHAPE
    action_count: int = len(PILOT_ACTIONS)
    include_bias: bool = True
    gamma: float = 0.94
    learning_rate: float = 0.03
    l2: float = 0.0005
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 25_000
    replay_capacity: int = 20_000
    min_replay_size: int = 64
    batch_size: int = 32
    target_sync_interval: int = 200
    gradient_steps: int = 1
    seed_scale: float = 0.01

    @property
    def feature_count(self) -> int:
        return int(np.prod(self.state_shape)) + (1 if self.include_bias else 0)


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


def normalize_state(state: np.ndarray | Iterable[float], shape: tuple[int, ...] = FRAME_SHAPE) -> np.ndarray:
    """Return one state as float32 with the configured shape."""

    array = np.asarray(state, dtype=np.float32)
    if array.shape == shape:
        return array
    if shape == FRAME_SHAPE and array.shape == FRAME_SHAPE[:2]:
        return array[..., np.newaxis]
    if shape == FRAME_SHAPE and array.shape == (FLAT_FRAME_SIZE,):
        return array.reshape(FRAME_SHAPE)
    if len(shape) == 1 and array.shape == (shape[0],):
        return array
    raise ValueError(f"Expected state shape {shape}, got {array.shape}.")


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
    """Map the enemy agent's discrete action to the Pygame fleet move interface."""

    moves = {
        0: [-1, 1],
        1: [0, 1],
        2: [1, 1],
        3: [0, 1],
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
    """Linear Double-DQN agent with replay, target weights, and checkpoints."""

    def __init__(
        self,
        *,
        config: DQNConfig | None = None,
        weights_path: Path | str | None = None,
        name: str = "pilot",
        seed: int | None = None,
        load_existing: bool = True,
        warn_legacy: bool = True,
    ):
        self.config = config or DQNConfig()
        self.name = name
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_PILOT_WEIGHTS
        self.random = random.Random(seed)
        self.rng = np.random.default_rng(seed)
        self.replay = ReplayBuffer(self.config.replay_capacity)
        self.training_steps = 0
        self.episodes = 0
        self.last_loss: float | None = None
        self.weights = self.rng.normal(
            0.0,
            self.config.seed_scale,
            size=(self.config.feature_count, self.config.action_count),
        ).astype(np.float32)
        self.target_weights = self.weights.copy()

        if load_existing and self.weights_path.exists():
            self.load(self.weights_path)
        elif warn_legacy and LEGACY_WEIGHTS.exists() and self.name == "pilot":
            print(
                f"Legacy TensorFlow weights found at {LEGACY_WEIGHTS}; NumPy DQN "
                f"checkpoints use {self.weights_path}."
            )

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.training_steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + progress * (self.config.epsilon_final - self.config.epsilon_start)

    def features(self, state: np.ndarray | Iterable[float]) -> np.ndarray:
        state_array = normalize_state(state, self.config.state_shape).reshape(-1).astype(np.float32)
        if self.config.include_bias:
            return np.concatenate(([1.0], state_array)).astype(np.float32)
        return state_array

    def q_values(self, state: np.ndarray | Iterable[float], *, target: bool = False) -> np.ndarray:
        weights = self.target_weights if target else self.weights
        return self.features(state) @ weights

    def act(self, state: np.ndarray, *, training: bool = True) -> int:
        if training and self.random.random() < self.epsilon:
            return self.random.randrange(self.config.action_count)
        return int(np.argmax(self.q_values(state)))

    def remember(self, state, action, reward, next_state, done) -> None:
        self.replay.append(
            Transition(
                normalize_state(state, self.config.state_shape),
                action_from_vector(action),
                float(reward),
                normalize_state(next_state, self.config.state_shape),
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
            for item in batch:
                features = self.features(item.state)
                next_features = self.features(item.next_state)
                prediction = float(features @ self.weights[:, item.action])
                next_action = int(np.argmax(next_features @ self.weights))
                next_value = float(next_features @ self.target_weights[:, next_action])
                target = item.reward if item.done else item.reward + self.config.gamma * next_value
                td_error = prediction - target
                huber_grad = td_error if abs(td_error) <= 1.0 else np.sign(td_error)
                self.weights[:, item.action] -= self.config.learning_rate * (
                    huber_grad * features + self.config.l2 * self.weights[:, item.action]
                )
                losses.append(0.5 * td_error * td_error if abs(td_error) <= 1.0 else abs(td_error) - 0.5)

            self.training_steps += 1
            if self.training_steps % self.config.target_sync_interval == 0:
                self.sync_target_network()

        self.last_loss = float(np.mean(losses))
        return self.last_loss

    def sync_target_network(self) -> None:
        self.target_weights = self.weights.copy()

    def save(self, path: Path | str | None = None) -> Path:
        destination = Path(path) if path else self.weights_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            destination,
            weights=self.weights,
            target_weights=self.target_weights,
            training_steps=np.asarray([self.training_steps], dtype=np.int64),
            action_count=np.asarray([self.config.action_count], dtype=np.int64),
            include_bias=np.asarray([int(self.config.include_bias)], dtype=np.int64),
            state_shape=np.asarray(self.config.state_shape, dtype=np.int64),
        )
        return destination

    def load(self, path: Path | str) -> None:
        payload = np.load(Path(path), allow_pickle=False)
        weights = payload["weights"].astype(np.float32)
        if weights.shape != self.weights.shape:
            raise ValueError(f"Checkpoint shape {weights.shape} does not match agent shape {self.weights.shape}.")
        self.weights = weights
        self.target_weights = (
            payload["target_weights"].astype(np.float32) if "target_weights" in payload.files else weights.copy()
        )
        self.training_steps = int(payload["training_steps"][0]) if "training_steps" in payload.files else 0

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
            weights_path=pilot_weights_path or self.checkpoint_dir / "pilot.npz",
            name="pilot",
            seed=base_seed,
        )
        self.enemy = DQNAgent(
            config=DQNConfig(action_count=len(ENEMY_ACTIONS)),
            weights_path=enemy_weights_path or self.checkpoint_dir / "enemy.npz",
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
        path = agent.save(self.checkpoint_dir / f"{role}-episode-{episode:05d}.npz")
        agent.save()
        return SelfPlayCheckpoint(role=role, episode=episode, weights_path=path, score=score, epsilon=agent.epsilon)
