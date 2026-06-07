"""Train RL pilot and enemy policies for the static GalagAI demo.

The trainer uses a headless Gymnasium version of the browser game loop and
alternates Stable-Baselines3 DQN training between the pilot and enemy roles. The
exported JSON contains both learned Q networks so GitHub Pages can run the
latest pilot and enemy policies without loading Python dependencies.
"""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm.auto import tqdm


PILOT_ACTIONS = ["left", "right", "fire", "stay"]
ENEMY_ACTIONS = [
    "drift_left",
    "drift_left_fire",
    "drop",
    "drift_right",
    "drift_right_fire",
    "fire",
    "dive",
    "loop",
    "scatter",
]
FEATURES = [
    "target_dx",
    "abs_target_dx",
    "threat_dx",
    "threat_y",
    "bullet_ready",
    "alien_count",
    "wave",
    "drop_ready",
    "enemy_shot_ready",
    "fleet_y",
    "lives",
]

CANVAS_WIDTH = 960.0
CANVAS_HEIGHT = 560.0
SHIP_WIDTH = 64.0
SHIP_HEIGHT = 48.0
SHIP_Y = CANVAS_HEIGHT - 72.0
SHIP_SPEED = 470.0
BULLET_SPEED = 620.0
ENEMY_SHOT_SPEED = 230.0
ALIEN_WIDTH = 48.0
ALIEN_HEIGHT = 34.0
FLEET_DROP = 18.0
MAX_ALIENS_NORMALIZER = 45.0
ACTION_DT = 0.12
DROP_COOLDOWN_SECONDS = 1.08
ENEMY_SHOT_COOLDOWN_SECONDS = 0.0
INVALID_DROP_PENALTY = 0.90
MODEL_SCHEMA_VERSION = 10
DQN_NET_ARCH = [64, 64]
MODEL_FILE_DIR = "galagai-models"
DEFAULT_CHECKPOINT_DIR = Path(".training-checkpoints/galagai")


@dataclass
class Actor:
    x: float
    y: float
    width: float
    height: float
    alive: bool = True
    role: str = "bee"

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass
class Shot:
    x: float
    y: float
    width: float
    height: float
    speed: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0


@dataclass(frozen=True)
class StepEvents:
    score_gain: int
    aliens_destroyed: int
    life_loss: int
    wave_cleared: bool
    invalid_drop: bool
    invalid_fire: bool
    valid_drop: bool
    enemy_fired: bool
    pilot_fired: bool
    pilot_missed: bool
    pilot_hits: int
    pilot_aligned_action: bool
    pilot_aligned_fire: bool
    pilot_bad_fire: bool
    enemy_tactical_action: bool


@dataclass(frozen=True)
class EpisodeResult:
    winner: str
    score: int
    wave: int
    steps: int
    enemy_drop_rate: float
    invalid_drop_rate: float
    enemy_fire_rate: float
    pilot_fire_rate: float
    pilot_hits: int
    pilot_fires: int
    pilot_shot_accuracy: float
    wave_cleared: bool
    lives_left: int


@dataclass(frozen=True)
class LoadedTrainingState:
    pilot_model: DQN | None
    enemy_model: DQN | None
    history: list[dict[str, object]]
    checkpoints: dict[str, list[dict[str, object]]]
    round_number: int
    phase_number: int


PolicySpec = dict[str, object]


class Policy(Protocol):
    def act(self, observation: np.ndarray) -> int:
        ...


class HeuristicPilotPolicy:
    def act(self, observation: np.ndarray) -> int:
        target_dx = float(observation[0])
        threat_dx = float(observation[2])
        threat_y = float(observation[3])
        bullet_ready = observation[4] > 0.5
        if threat_y > 0.74 and abs(threat_dx) < 0.18:
            return 0 if threat_dx >= 0 else 1
        if bullet_ready and abs(target_dx) < 0.10:
            return 2
        if target_dx < -0.08:
            return 0
        if target_dx > 0.08:
            return 1
        return 3


class HeuristicEnemyPolicy:
    def act(self, observation: np.ndarray) -> int:
        target_dx = float(observation[0])
        drop_ready = observation[7] > 0.5
        fleet_y = float(observation[9])
        if abs(target_dx) < 0.18:
            return 5
        if drop_ready and fleet_y < 0.42 and abs(target_dx) > 0.34:
            return 2
        return 1 if target_dx > 0 else 4


class OpeningEnemyPolicy:
    def act(self, observation: np.ndarray) -> int:
        target_dx = float(observation[0])
        return 0 if target_dx > 0 else 3


class SB3Policy:
    def __init__(self, model: DQN | None):
        if model is None:
            raise ValueError("SB3Policy requires a loaded DQN model.")
        self.model = model

    def act(self, observation: np.ndarray) -> int:
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)


class NetworkPolicy:
    """Run an exported Q-network without keeping a live SB3 model in a worker."""

    def __init__(self, network: dict[str, object]):
        self.activation = str(network.get("activation", "relu"))
        self.layers = list(network.get("layers", []))

    def act(self, observation: np.ndarray) -> int:
        values = np.asarray(observation, dtype=np.float32)
        for index, layer in enumerate(self.layers):
            weights = np.asarray(layer["weights"], dtype=np.float32)
            biases = np.asarray(layer["biases"], dtype=np.float32)
            values = values @ weights + biases
            if index < len(self.layers) - 1 and self.activation == "relu":
                values = np.maximum(values, 0.0)
        return int(np.argmax(values)) if values.size else 0


def pilot_policy_spec(model: DQN | None) -> PolicySpec:
    if model is None:
        return {"kind": "heuristic-pilot"}
    return {"kind": "network", "role": "pilot", "network": export_network(model)}


def enemy_policy_spec(model: DQN | None) -> PolicySpec:
    if model is None:
        return {"kind": "opening-enemy"}
    return {"kind": "network", "role": "enemies", "network": export_network(model)}


def policy_from_spec(spec: PolicySpec) -> Policy:
    kind = str(spec["kind"])
    if kind == "network":
        network = spec.get("network")
        if not isinstance(network, dict):
            raise ValueError("Network policy spec is missing its exported network.")
        return NetworkPolicy(network)
    if kind == "heuristic-pilot":
        return HeuristicPilotPolicy()
    if kind == "heuristic-enemy":
        return HeuristicEnemyPolicy()
    if kind == "opening-enemy":
        return OpeningEnemyPolicy()
    raise ValueError(f"Unknown policy spec kind {kind}.")


class HeadlessGalagai:
    """Browser-like GalagAI transition loop for RL training."""

    def __init__(self, seed: int = 0, max_steps: int = 520):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.score = 0
        self.wave = 1
        self.lives = 3
        self.steps = 0
        self.ship = Actor(CANVAS_WIDTH / 2.0 - SHIP_WIDTH / 2.0, SHIP_Y, SHIP_WIDTH, SHIP_HEIGHT)
        self.bullets: list[Shot] = []
        self.enemy_shots: list[Shot] = []
        self.fire_cooldown = 0.0
        self.enemy_shot_cooldown = 0.0
        self.enemy_drop_cooldown = 0.0
        self.invulnerability = 0.0
        self.fleet_direction = 1
        self.fleet_speed = 38.0
        self.aliens = self.create_fleet(self.wave)
        self.drop_attempts = 0
        self.invalid_drops = 0
        self.enemy_fires = 0
        self.pilot_fires = 0
        self.pilot_hits = 0
        return self.features()

    def create_fleet(self, wave: int) -> list[Actor]:
        aliens: list[Actor] = []
        columns = min(9, 6 + wave)
        rows = min(5, 3 + wave // 2)
        gap_x = 78.0
        gap_y = 54.0
        start_x = (CANVAS_WIDTH - (columns - 1) * gap_x) / 2.0 - 24.0
        for row in range(rows):
            for col in range(columns):
                aliens.append(
                    Actor(
                        start_x + col * gap_x,
                        74.0 + row * gap_y,
                        ALIEN_WIDTH,
                        ALIEN_HEIGHT,
                        role=self.enemy_role_for_slot(row, col, wave),
                    )
                )
        return aliens

    @staticmethod
    def enemy_role_for_slot(row: int, col: int, wave: int) -> str:
        if wave >= 3 and row == 0 and col % 3 == 1:
            return "boss"
        if wave >= 2 and row <= 1 and col % 2 == 0:
            return "butterfly"
        return "bee"

    def features(self) -> np.ndarray:
        ship_center = self.ship.center_x
        live_aliens = [alien for alien in self.aliens if alien.alive]
        target = min(live_aliens, key=lambda alien: abs(alien.center_x - ship_center), default=None)
        target_dx = self._relative_x(target.center_x - ship_center) if target else 0.0

        closest_shot = min(self.enemy_shots, key=lambda shot: abs(shot.center_x - ship_center), default=None)
        threat_dx = self._relative_x(closest_shot.center_x - ship_center) if closest_shot else 1.0
        threat_y = self._clamp01(closest_shot.y / CANVAS_HEIGHT) if closest_shot else 0.0
        fleet_y = self._clamp01(max((alien.y for alien in live_aliens), default=0.0) / CANVAS_HEIGHT)

        return np.asarray(
            [
                target_dx,
                abs(target_dx),
                threat_dx,
                threat_y,
                1.0 if self.fire_cooldown <= 0 else 0.0,
                self._clamp01(len(live_aliens) / MAX_ALIENS_NORMALIZER),
                self._clamp01(self.wave / 10.0),
                1.0 if self.enemy_drop_cooldown <= 0 else 0.0,
                1.0,
                fleet_y,
                self._clamp01(self.lives / 3.0),
            ],
            dtype=np.float32,
        )

    def step(self, pilot_action: int, enemy_action: int) -> tuple[np.ndarray, float, float, bool, dict[str, object]]:
        self.steps += 1
        before_score = self.score
        before_lives = self.lives
        before_aliens = self.live_alien_count
        pilot_fired = False
        pilot_missed = False
        enemy_fired = False
        invalid_drop = False
        invalid_fire = False
        valid_drop = False

        self.fire_cooldown = max(0.0, self.fire_cooldown - ACTION_DT)
        self.enemy_shot_cooldown = max(0.0, self.enemy_shot_cooldown - ACTION_DT)
        self.enemy_drop_cooldown = max(0.0, self.enemy_drop_cooldown - ACTION_DT)
        self.invulnerability = max(0.0, self.invulnerability - ACTION_DT)

        action_features = self.features()
        target_dx = float(action_features[0])
        fleet_y = float(action_features[9])
        pilot_aligned_action = self.is_pilot_aligned_action(pilot_action, target_dx)
        pilot_aligned_fire = pilot_action == 2 and self.fire_cooldown <= 0 and abs(target_dx) < 0.16
        pilot_bad_fire = pilot_action == 2 and self.fire_cooldown <= 0 and abs(target_dx) > 0.30
        enemy_tactical_action = self.is_enemy_tactical_action(enemy_action, target_dx, fleet_y)

        if pilot_action == 0:
            self.ship.x -= SHIP_SPEED * ACTION_DT
        elif pilot_action == 1:
            self.ship.x += SHIP_SPEED * ACTION_DT
        elif pilot_action == 2:
            pilot_fired, pilot_missed = self.fire_pilot_bullet()
        elif pilot_action != 3:
            raise ValueError(f"Unknown pilot action {pilot_action}.")
        self.ship.x = self._clamp(self.ship.x, 18.0, CANVAS_WIDTH - SHIP_WIDTH - 18.0)

        if enemy_action == 0:
            self.fleet_direction = -1
        elif enemy_action == 1:
            self.fleet_direction = -1
            enemy_fired = self.fire_enemy_shot(("butterfly", "boss"))
            if enemy_fired:
                self.enemy_fires += 1
            else:
                invalid_fire = True
        elif enemy_action == 2:
            self.drop_attempts += 1
            if self.enemy_drop_cooldown <= 0:
                self.drop_fleet()
                self.enemy_drop_cooldown = DROP_COOLDOWN_SECONDS
                valid_drop = True
            else:
                self.invalid_drops += 1
                invalid_drop = True
        elif enemy_action == 3:
            self.fleet_direction = 1
        elif enemy_action == 4:
            self.fleet_direction = 1
            enemy_fired = self.fire_enemy_shot(("butterfly", "boss"))
            if enemy_fired:
                self.enemy_fires += 1
            else:
                invalid_fire = True
        elif enemy_action == 5:
            enemy_fired = self.fire_enemy_shot(("butterfly", "boss"))
            if enemy_fired:
                self.enemy_fires += 1
            else:
                invalid_fire = True
        elif enemy_action == 6:
            self.fleet_direction = -1 if target_dx < 0 else 1
            self.dive_fleet(("butterfly", "boss"))
            enemy_fired = self.fire_enemy_shot(("butterfly", "boss"))
            if enemy_fired:
                self.enemy_fires += 1
            else:
                invalid_fire = True
        elif enemy_action == 7:
            self.fleet_direction *= -1
            enemy_fired = self.fire_enemy_shot(("boss",))
            if enemy_fired:
                self.enemy_fires += 1
            else:
                invalid_fire = True
        elif enemy_action == 8:
            self.fleet_direction = 1 if target_dx < 0 else -1
        else:
            raise ValueError(f"Unknown enemy action {enemy_action}.")

        self.update_projectiles()
        self.update_aliens()
        hit_events = self.resolve_collisions()
        pilot_hits = int(hit_events["aliensHit"])

        done = False
        wave_cleared = False
        if self.live_alien_count == 0:
            self.wave += 1
            self.score += 250
            self.fleet_speed += 13.0
            self.fleet_direction = 1
            self.aliens = self.create_fleet(self.wave)
            self.bullets.clear()
            wave_cleared = True
        if self.lives <= 0:
            done = True
        elif self.steps >= self.max_steps:
            done = True

        events = StepEvents(
            score_gain=self.score - before_score,
            aliens_destroyed=before_aliens - self.live_alien_count,
            life_loss=before_lives - self.lives,
            wave_cleared=wave_cleared,
            invalid_drop=invalid_drop,
            invalid_fire=invalid_fire,
            valid_drop=valid_drop,
            enemy_fired=enemy_fired,
            pilot_fired=pilot_fired,
            pilot_missed=pilot_missed,
            pilot_hits=pilot_hits,
            pilot_aligned_action=pilot_aligned_action,
            pilot_aligned_fire=pilot_aligned_fire,
            pilot_bad_fire=pilot_bad_fire,
            enemy_tactical_action=enemy_tactical_action,
        )
        pilot_reward = self.pilot_reward(events, done)
        enemy_reward = self.enemy_reward(events, done)
        info = {
            "events": events,
            "score": self.score,
            "wave": self.wave,
            "lives": self.lives,
            "winner": self.winner(done),
            "dropAttempts": self.drop_attempts,
            "invalidDrops": self.invalid_drops,
            "enemyFires": self.enemy_fires,
            "pilotFires": self.pilot_fires,
            "pilotHits": self.pilot_hits,
        }
        return self.features(), pilot_reward, enemy_reward, done, info

    @property
    def live_alien_count(self) -> int:
        return sum(1 for alien in self.aliens if alien.alive)

    def fire_pilot_bullet(self) -> tuple[bool, bool]:
        if self.fire_cooldown > 0:
            return False, True
        self.bullets.append(Shot(self.ship.center_x - 3.0, self.ship.y - 14.0, 6.0, 18.0, BULLET_SPEED))
        self.fire_cooldown = 0.17
        self.pilot_fires += 1
        return True, False

    def fire_enemy_shot(self, roles: tuple[str, ...] | None = None) -> bool:
        live_aliens = [
            alien
            for alien in self.aliens
            if alien.alive and (roles is None or alien.role in roles)
        ]
        if not live_aliens:
            return False
        alien = min(live_aliens, key=lambda item: abs(item.center_x - self.ship.center_x))
        self.enemy_shots.append(Shot(alien.center_x - 3.0, alien.bottom, 6.0, 16.0, ENEMY_SHOT_SPEED + self.wave * 20.0))
        return True

    def drop_fleet(self) -> None:
        for alien in self.aliens:
            if alien.alive and alien.role == "bee":
                alien.y += FLEET_DROP * 0.65

    def dive_fleet(self, roles: tuple[str, ...] | None = None) -> None:
        live_aliens = [
            alien
            for alien in self.aliens
            if alien.alive and (roles is None or alien.role in roles)
        ]
        if not live_aliens:
            return
        diver = min(live_aliens, key=lambda alien: abs(alien.center_x - self.ship.center_x))
        diver.x += self._clamp(self.ship.center_x - diver.center_x, -28.0, 28.0)
        diver.y += FLEET_DROP * 0.42

    def update_projectiles(self) -> None:
        for bullet in self.bullets:
            bullet.y -= bullet.speed * ACTION_DT
        for shot in self.enemy_shots:
            shot.y += shot.speed * ACTION_DT
        self.bullets = [bullet for bullet in self.bullets if bullet.y > -bullet.height]
        self.enemy_shots = [shot for shot in self.enemy_shots if shot.y < CANVAS_HEIGHT + shot.height]

    def update_aliens(self) -> None:
        live_aliens = [alien for alien in self.aliens if alien.alive]
        if not live_aliens:
            return
        hit_edge = any(
            alien.x + self.fleet_direction * self.fleet_speed * ACTION_DT < 16.0
            or alien.x + ALIEN_WIDTH + self.fleet_direction * self.fleet_speed * ACTION_DT > CANVAS_WIDTH - 16.0
            for alien in live_aliens
        )
        if hit_edge:
            self.fleet_direction *= -1
            for alien in live_aliens:
                alien.y += FLEET_DROP
        for alien in live_aliens:
            alien.x += self.fleet_direction * self.fleet_speed * ACTION_DT
            if alien.bottom >= self.ship.y and self.invulnerability <= 0:
                self.lose_life()

    def resolve_collisions(self) -> dict[str, int]:
        aliens_hit = 0
        for bullet in self.bullets:
            for alien in self.aliens:
                if alien.alive and self.intersects(bullet, alien):
                    alien.alive = False
                    bullet.y = -100.0
                    aliens_hit += 1
                    self.pilot_hits += 1
                    self.score += 50 * self.wave
                    break

        ship_hits = 0
        for shot in self.enemy_shots:
            if self.intersects(shot, self.ship) and self.invulnerability <= 0:
                shot.y = CANVAS_HEIGHT + 100.0
                ship_hits += 1
                self.lose_life()
        return {"aliensHit": aliens_hit, "shipHits": ship_hits}

    def lose_life(self) -> None:
        if self.invulnerability > 0:
            return
        self.lives -= 1
        self.invulnerability = 0.45
        self.enemy_shots.clear()
        self.ship.x = CANVAS_WIDTH / 2.0 - SHIP_WIDTH / 2.0

    def pilot_reward(self, events: StepEvents, done: bool) -> float:
        reward = events.score_gain / 35.0
        clear_speed = max(0.0, 1.0 - self.steps / max(1, self.max_steps))
        reward += 8.0 + 6.0 * clear_speed if events.wave_cleared else 0.0
        reward += 1.15 * max(0, events.pilot_hits)
        reward -= 7.0 * max(0, events.life_loss)
        reward += 0.08 if events.pilot_aligned_action else 0.0
        if events.pilot_fired:
            reward += 0.42 if events.pilot_aligned_fire else -0.22
        reward -= 0.48 if events.pilot_bad_fire else 0.0
        reward -= 0.32 if events.pilot_missed else 0.0
        reward -= 0.008
        if done:
            winner = self.winner(done)
            if winner == "pilot":
                reward += 6.0 + 3.0 * clear_speed
            elif winner == "enemies":
                reward -= 6.0
            else:
                reward -= 1.0
        return float(reward)

    def enemy_reward(self, events: StepEvents, done: bool) -> float:
        clear_speed = max(0.0, 1.0 - self.steps / max(1, self.max_steps))
        reward = 7.0 * max(0, events.life_loss)
        reward -= 1.45 * max(0, events.aliens_destroyed)
        reward -= events.score_gain / 55.0
        reward -= 2.0 + 2.0 * clear_speed if events.wave_cleared else 0.0
        reward += 0.02 * self.live_alien_count / max(1, len(self.aliens))
        reward += 0.12 if events.enemy_tactical_action else 0.0
        if events.valid_drop:
            reward += 0.04 if events.enemy_tactical_action else -0.10
        reward -= INVALID_DROP_PENALTY if events.invalid_drop else 0.0
        reward -= 0.25 if events.invalid_fire else 0.0
        reward += 0.05 if events.enemy_fired else 0.0
        if done:
            winner = self.winner(done)
            if winner == "enemies":
                reward += 6.0
            elif winner == "pilot":
                reward -= 5.0
            else:
                reward += 0.5
        return float(reward)

    def winner(self, done: bool) -> str:
        if not done:
            return "none"
        if self.lives <= 0:
            return "enemies"
        if self.live_alien_count == 0 or self.wave > 1:
            return "pilot"
        return "timeout"

    @staticmethod
    def is_pilot_aligned_action(action: int, target_dx: float) -> bool:
        if action == 0:
            return target_dx < -0.06
        if action == 1:
            return target_dx > 0.06
        if action == 2:
            return abs(target_dx) < 0.16
        if action == 3:
            return abs(target_dx) < 0.05
        return False

    @staticmethod
    def is_enemy_tactical_action(action: int, target_dx: float, fleet_y: float) -> bool:
        if action == 0:
            return target_dx > 0.06
        if action == 1:
            return target_dx > 0.06 or abs(target_dx) < 0.18
        if action == 2:
            return fleet_y < 0.42 and abs(target_dx) > 0.34
        if action == 3:
            return target_dx < -0.06
        if action == 4:
            return target_dx < -0.06 or abs(target_dx) < 0.18
        if action == 5:
            return abs(target_dx) < 0.18
        if action == 6:
            return abs(target_dx) < 0.32
        if action == 7:
            return True
        if action == 8:
            return abs(target_dx) < 0.12
        return False

    @staticmethod
    def intersects(a: Actor | Shot, b: Actor | Shot) -> bool:
        return a.x < b.x + b.width and a.x + a.width > b.x and a.y < b.y + b.height and a.y + a.height > b.y

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _relative_x(value: float) -> float:
        return max(-1.0, min(1.0, value / (CANVAS_WIDTH / 2.0)))


class PilotTrainingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent: Policy, seed: int = 0, max_steps: int = 520):
        super().__init__()
        self.opponent = opponent
        self.game = HeadlessGalagai(seed=seed, max_steps=max_steps)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(len(FEATURES),), dtype=np.float32)
        self.action_space = spaces.Discrete(len(PILOT_ACTIONS))
        self.seed_value = seed

    def reset(self, *, seed: int | None = None, options=None):
        self.seed_value = self.seed_value + 1 if seed is None else seed
        return self.game.reset(seed=self.seed_value), {}

    def step(self, action: int):
        observation = self.game.features()
        enemy_action = self.opponent.act(observation)
        next_observation, pilot_reward, _, done, info = self.game.step(int(action), enemy_action)
        terminated = done and str(info.get("winner", "none")) in {"pilot", "enemies"}
        truncated = done and not terminated
        return next_observation, pilot_reward, terminated, truncated, serializable_info(info)


class EnemyTrainingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent: Policy, seed: int = 0, max_steps: int = 520):
        super().__init__()
        self.opponent = opponent
        self.game = HeadlessGalagai(seed=seed, max_steps=max_steps)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(len(FEATURES),), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ENEMY_ACTIONS))
        self.seed_value = seed

    def reset(self, *, seed: int | None = None, options=None):
        self.seed_value = self.seed_value + 1 if seed is None else seed
        return self.game.reset(seed=self.seed_value), {}

    def step(self, action: int):
        observation = self.game.features()
        pilot_action = self.opponent.act(observation)
        next_observation, _, enemy_reward, done, info = self.game.step(pilot_action, int(action))
        terminated = done and str(info.get("winner", "none")) in {"pilot", "enemies"}
        truncated = done and not terminated
        return next_observation, enemy_reward, terminated, truncated, serializable_info(info)


def serializable_info(info: dict[str, object]) -> dict[str, object]:
    sanitized = dict(info)
    events = sanitized.get("events")
    if isinstance(events, StepEvents):
        sanitized["events"] = {
            "scoreGain": events.score_gain,
            "aliensDestroyed": events.aliens_destroyed,
            "lifeLoss": events.life_loss,
            "waveCleared": events.wave_cleared,
            "invalidDrop": events.invalid_drop,
            "invalidFire": events.invalid_fire,
            "validDrop": events.valid_drop,
            "enemyFired": events.enemy_fired,
            "pilotFired": events.pilot_fired,
            "pilotMissed": events.pilot_missed,
            "pilotHits": events.pilot_hits,
            "pilotAlignedAction": events.pilot_aligned_action,
            "pilotAlignedFire": events.pilot_aligned_fire,
            "pilotBadFire": events.pilot_bad_fire,
            "enemyTacticalAction": events.enemy_tactical_action,
        }
    return sanitized


def make_role_env(role: str, opponent_spec: PolicySpec, seed: int, max_steps: int) -> gym.Env:
    opponent = policy_from_spec(opponent_spec)
    if role == "pilot":
        return Monitor(PilotTrainingEnv(opponent, seed=seed, max_steps=max_steps))
    if role == "enemies":
        return Monitor(EnemyTrainingEnv(opponent, seed=seed, max_steps=max_steps))
    raise ValueError(f"Unknown training role {role}.")


def make_training_env(
    role: str,
    opponent_spec: PolicySpec,
    *,
    seed: int,
    max_steps: int,
    workers: int,
) -> Any:
    worker_count = max(1, int(workers))
    if worker_count == 1:
        return make_role_env(role, opponent_spec, seed=seed, max_steps=max_steps)

    def make_env(worker_index: int):
        def _init() -> gym.Env:
            return make_role_env(
                role,
                opponent_spec,
                seed=seed + worker_index * 10_000,
                max_steps=max_steps,
            )

        return _init

    return SubprocVecEnv([make_env(index) for index in range(worker_count)], start_method="spawn")


def close_training_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except (BrokenPipeError, EOFError):
            pass


def make_dqn(env: Any, seed: int, learning_rate: float) -> DQN:
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=50_000,
        learning_starts=250,
        batch_size=64,
        gamma=0.97,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": DQN_NET_ARCH},
        seed=seed,
        verbose=0,
    )


class TrainingProgress:
    def __init__(self, enabled: bool, total: int, completed: int = 0):
        self.bar = tqdm(
            total=total,
            desc="self-play generations",
            unit="gen",
            dynamic_ncols=True,
            disable=not enabled,
        )
        if completed and not self.bar.disable:
            self.bar.update(min(completed, total))

    def update(self, metrics: dict[str, object], pilot_count: int, enemy_count: int) -> None:
        if self.bar.disable:
            return
        self.bar.set_postfix(
            side=metrics["trained"],
            pilot=pilot_count,
            enemies=enemy_count,
            p_win=f"{float(metrics['pilotWinRate']):.2f}",
            e_win=f"{float(metrics['enemyWinRate']):.2f}",
            acc=f"{float(metrics.get('pilotShotAccuracy', 0.0)):.2f}",
            clear=f"{float(metrics.get('waveClearRate', 0.0)):.2f}",
            drop=f"{float(metrics['enemyDropRate']):.3f}",
            invalid=f"{float(metrics['invalidDropRate']):.3f}",
            reached=str(bool(metrics["dominanceReached"])).lower(),
            refresh=False,
        )
        self.bar.update(1)

    def write(self, message: str) -> None:
        if not self.bar.disable:
            self.bar.write(message)

    def close(self) -> None:
        self.bar.close()


def progress_total(
    cycles: int,
    max_phase_iterations: int,
    generations_per_side: int | None,
    rounds: int | None,
    balanced_rounds: int | None,
) -> int:
    if balanced_rounds is not None:
        return balanced_rounds
    if generations_per_side is not None:
        return generations_per_side * 2
    if rounds is not None:
        return rounds
    return cycles * 2 * max_phase_iterations


def balanced_metric_reached(
    metrics: dict[str, object],
    *,
    dominance_threshold: float,
    balance_tolerance: float,
    balance_min_win_rate: float,
) -> bool:
    pilot_win_rate = float(metrics.get("pilotWinRate", 0.0))
    enemy_win_rate = float(metrics.get("enemyWinRate", 0.0))
    return (
        max(pilot_win_rate, enemy_win_rate) < dominance_threshold
        and max(pilot_win_rate, enemy_win_rate) >= balance_min_win_rate
        and abs(pilot_win_rate - enemy_win_rate) <= balance_tolerance
    )


def balanced_stop_reached(
    history: list[dict[str, object]],
    *,
    min_balanced_rounds: int,
    balance_patience: int,
    dominance_threshold: float,
    balance_tolerance: float,
    balance_min_win_rate: float,
) -> bool:
    patience = max(1, balance_patience)
    if len(history) < max(min_balanced_rounds, patience):
        return False
    return all(
        balanced_metric_reached(
            metrics,
            dominance_threshold=dominance_threshold,
            balance_tolerance=balance_tolerance,
            balance_min_win_rate=balance_min_win_rate,
        )
        for metrics in history[-patience:]
    )


def choose_balanced_role(
    latest_metrics: dict[str, object] | None,
    checkpoints: dict[str, list[dict[str, object]]],
    *,
    dominance_threshold: float,
) -> str:
    if not checkpoints["pilot"]:
        return "pilot"
    if not checkpoints["enemies"]:
        return "enemies"
    if latest_metrics is None:
        return "pilot"

    pilot_win_rate = float(latest_metrics.get("pilotWinRate", 0.0))
    enemy_win_rate = float(latest_metrics.get("enemyWinRate", 0.0))
    if enemy_win_rate >= dominance_threshold and enemy_win_rate >= pilot_win_rate:
        return "pilot"
    if pilot_win_rate >= dominance_threshold and pilot_win_rate > enemy_win_rate:
        return "enemies"
    return "pilot" if pilot_win_rate <= enemy_win_rate else "enemies"


class TrainingCheckpointStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.exports_dir = directory / "exports"
        self.state_path = directory / "state.json"

    def has_state(self) -> bool:
        return self.state_path.exists()

    def load(self) -> LoadedTrainingState:
        if not self.state_path.exists():
            raise RuntimeError(f"No training checkpoint found at {self.state_path}.")

        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self._validate_state(state)
        history = list(state.get("rounds", []))
        checkpoint_files = state.get("checkpointFiles", {})
        checkpoints = {
            "pilot": self._load_exported_entries("pilot", checkpoint_files),
            "enemies": self._load_exported_entries("enemies", checkpoint_files),
        }
        return LoadedTrainingState(
            pilot_model=self._load_role_model("pilot") if checkpoints["pilot"] else None,
            enemy_model=self._load_role_model("enemies") if checkpoints["enemies"] else None,
            history=history,
            checkpoints=checkpoints,
            round_number=int(state.get("roundNumber", len(history))),
            phase_number=int(state.get("phaseNumber", self._latest_phase(history))),
        )

    def save(
        self,
        *,
        pilot_model: DQN | None,
        enemy_model: DQN | None,
        history: list[dict[str, object]],
        checkpoints: dict[str, list[dict[str, object]]],
        round_number: int,
        phase_number: int,
        config: dict[str, object],
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        if pilot_model is not None:
            self._save_role_model("pilot", pilot_model)
        if enemy_model is not None:
            self._save_role_model("enemies", enemy_model)

        checkpoint_files: dict[str, list[str]] = {"pilot": [], "enemies": []}
        for role in ("pilot", "enemies"):
            entries = checkpoints.get(role, [])
            for entry in entries:
                filename = checkpoint_filename(role, int(entry["id"]))
                entry_path = self.exports_dir / filename
                checkpoint_files[role].append(filename)
                if not entry_path.exists() or int(entry["id"]) == len(entries):
                    self._write_json_atomic(entry_path, entry)

        state = {
            "schemaVersion": MODEL_SCHEMA_VERSION,
            "algorithm": "stable-baselines3-dqn",
            "actions": {"pilot": PILOT_ACTIONS, "enemies": ENEMY_ACTIONS},
            "features": FEATURES,
            "netArch": DQN_NET_ARCH,
            "roundNumber": round_number,
            "phaseNumber": phase_number,
            "checkpointCounts": {
                "pilot": len(checkpoints.get("pilot", [])),
                "enemies": len(checkpoints.get("enemies", [])),
            },
            "checkpointFiles": checkpoint_files,
            "rounds": history,
            "config": config,
        }
        self._write_json_atomic(self.state_path, state)

    def _validate_state(self, state: dict[str, object]) -> None:
        if int(state.get("schemaVersion", -1)) != MODEL_SCHEMA_VERSION:
            raise RuntimeError(
                f"Checkpoint schema {state.get('schemaVersion')} does not match current schema {MODEL_SCHEMA_VERSION}."
            )
        actions = state.get("actions", {})
        if not isinstance(actions, dict) or actions.get("pilot") != PILOT_ACTIONS or actions.get("enemies") != ENEMY_ACTIONS:
            raise RuntimeError("Checkpoint action space does not match the current trainer.")
        if state.get("features") != FEATURES:
            raise RuntimeError("Checkpoint feature vector does not match the current trainer.")
        if state.get("netArch") != DQN_NET_ARCH:
            raise RuntimeError("Checkpoint DQN network architecture does not match the current trainer.")

    def _load_exported_entries(self, role: str, checkpoint_files: object) -> list[dict[str, object]]:
        files: list[str] = []
        if isinstance(checkpoint_files, dict):
            role_files = checkpoint_files.get(role, [])
            if isinstance(role_files, list):
                files = [str(filename) for filename in role_files]
        entries = []
        for filename in files:
            path = self.exports_dir / filename
            if not path.exists():
                raise RuntimeError(f"Checkpoint export is missing: {path}")
            entries.append(json.loads(path.read_text(encoding="utf-8")))
        return entries

    def _load_role_model(self, role: str) -> DQN:
        model_path = self.directory / f"{role}_latest.zip"
        if not model_path.exists():
            raise RuntimeError(f"{role} model checkpoint is missing: {model_path}")
        model = DQN.load(model_path)
        replay_path = self.directory / f"{role}_replay.pkl"
        if replay_path.exists():
            model.load_replay_buffer(replay_path)
        return model

    def _save_role_model(self, role: str, model: DQN) -> None:
        self._save_model_atomic(model, self.directory / f"{role}_latest.zip")
        self._save_replay_buffer_atomic(model, self.directory / f"{role}_replay.pkl")

    @staticmethod
    def _save_model_atomic(model: DQN, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
        if tmp_path.exists():
            tmp_path.unlink()
        model.save(tmp_path)
        tmp_path.replace(path)

    @staticmethod
    def _save_replay_buffer_atomic(model: DQN, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
        if tmp_path.exists():
            tmp_path.unlink()
        model.save_replay_buffer(tmp_path)
        tmp_path.replace(path)

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
                json.dump(payload, tmp, separators=(",", ":"))
                tmp.write("\n")
                tmp_path = Path(tmp.name)
            tmp_path.replace(path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _latest_phase(history: list[dict[str, object]]) -> int:
        if not history:
            return 0
        return max(int(round_info.get("phase", 0)) for round_info in history)


def train_self_play(
    *,
    seed: int,
    cycles: int = 2,
    phase_timesteps: int = 2200,
    eval_episodes: int,
    max_steps: int,
    dominance_threshold: float = 0.6,
    max_phase_iterations: int = 3,
    generations_per_side: int | None = None,
    balanced_rounds: int | None = None,
    min_balanced_rounds: int = 6,
    balance_tolerance: float = 0.18,
    balance_patience: int = 3,
    balance_min_win_rate: float = 0.25,
    rounds: int | None = None,
    timesteps_per_round: int | None = None,
    progress: bool = False,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    train_workers: int = 1,
    eval_workers: int = 1,
) -> tuple[DQN, DQN, dict[str, object]]:
    if timesteps_per_round is not None:
        phase_timesteps = timesteps_per_round
    if balanced_rounds is not None and balanced_rounds < 2:
        raise ValueError("balanced_rounds must be at least 2 so both roles can get a checkpoint.")
    train_workers = max(1, int(train_workers))
    eval_workers = max(1, int(eval_workers))
    if balanced_rounds is not None:
        phase_roles = []
    elif generations_per_side is not None:
        phase_roles = []
    elif rounds is not None:
        phase_roles = ["pilot" if phase % 2 == 1 else "enemies" for phase in range(1, rounds + 1)]
        max_phase_iterations = 1
    else:
        phase_roles = [role for _ in range(cycles) for role in ("pilot", "enemies")]

    pilot_model: DQN | None = None
    enemy_model: DQN | None = None
    history: list[dict[str, object]] = []
    checkpoints: dict[str, list[dict[str, object]]] = {"pilot": [], "enemies": []}
    round_number = 0
    phase_number_offset = 0
    checkpoint_store = TrainingCheckpointStore(checkpoint_dir) if checkpoint_dir is not None else None
    if resume:
        if checkpoint_store is None:
            raise RuntimeError("--resume requires checkpoint_dir.")
        loaded = checkpoint_store.load()
        pilot_model = loaded.pilot_model
        enemy_model = loaded.enemy_model
        history = loaded.history
        checkpoints = loaded.checkpoints
        round_number = loaded.round_number
        phase_number_offset = loaded.phase_number

    completed_generations = len(history)
    progress_tracker = TrainingProgress(
        progress,
        progress_total(cycles, max_phase_iterations, generations_per_side, rounds, balanced_rounds),
        completed=completed_generations,
    )
    checkpoint_config = {
        "seed": seed,
        "cycles": cycles,
        "phaseTimesteps": phase_timesteps,
        "evalEpisodes": eval_episodes,
        "maxSteps": max_steps,
        "dominanceThreshold": dominance_threshold,
        "maxPhaseIterations": max_phase_iterations,
        "generationsPerSide": generations_per_side,
        "balancedRounds": balanced_rounds,
        "minBalancedRounds": min_balanced_rounds,
        "balanceTolerance": balance_tolerance,
        "balancePatience": balance_patience,
        "balanceMinWinRate": balance_min_win_rate,
        "rounds": rounds,
        "trainWorkers": train_workers,
        "evalWorkers": eval_workers,
    }

    def run_generation(role: str, phase_number: int, phase_iteration: int) -> bool:
        nonlocal pilot_model, enemy_model, round_number

        round_number += 1
        phase_seed = seed + phase_number * 1000 + phase_iteration * 97
        env = None
        if role == "pilot":
            env = make_training_env(
                "pilot",
                enemy_policy_spec(enemy_model),
                seed=phase_seed,
                max_steps=max_steps,
                workers=train_workers,
            )
            try:
                if pilot_model is None:
                    pilot_model = make_dqn(env, seed=seed, learning_rate=8e-4)
                else:
                    pilot_model.set_env(env)
                pilot_model.learn(total_timesteps=phase_timesteps, reset_num_timesteps=False, progress_bar=False)
            finally:
                close_training_env(env)
            trained = "pilot"
        else:
            env = make_training_env(
                "enemies",
                pilot_policy_spec(pilot_model),
                seed=phase_seed,
                max_steps=max_steps,
                workers=train_workers,
            )
            try:
                if enemy_model is None:
                    enemy_model = make_dqn(env, seed=seed + 1, learning_rate=8e-4)
                else:
                    enemy_model.set_env(env)
                enemy_model.learn(total_timesteps=phase_timesteps, reset_num_timesteps=False, progress_bar=False)
            finally:
                close_training_env(env)
            trained = "enemies"

        metrics = evaluate_current_matchup(
            seed=seed + round_number * 3000,
            pilot_model=pilot_model,
            enemy_model=enemy_model,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            eval_workers=eval_workers,
        )
        dominance_metric = "pilotWinRate" if trained == "pilot" else "enemyWinRate"
        dominance_reached = float(metrics[dominance_metric]) >= dominance_threshold
        round_metrics = {
            "round": round_number,
            "phase": phase_number,
            "phaseIteration": phase_iteration,
            "trained": trained,
            "generation": len(checkpoints[trained]) + 1,
            "dominanceMetric": dominance_metric,
            "dominanceThreshold": dominance_threshold,
            "dominanceReached": dominance_reached,
            **metrics,
        }
        history.append(round_metrics)

        if trained == "pilot" and pilot_model is not None:
            checkpoints["pilot"].append(
                checkpoint_entry(
                    role="pilot",
                    model=pilot_model,
                    version_id=len(checkpoints["pilot"]) + 1,
                    metrics=round_metrics,
                )
            )
        elif trained == "enemies" and enemy_model is not None:
            checkpoints["enemies"].append(
                checkpoint_entry(
                    role="enemies",
                    model=enemy_model,
                    version_id=len(checkpoints["enemies"]) + 1,
                    metrics=round_metrics,
                )
            )
        if checkpoint_store is not None:
            checkpoint_store.save(
                pilot_model=pilot_model,
                enemy_model=enemy_model,
                history=history,
                checkpoints=checkpoints,
                round_number=round_number,
                phase_number=phase_number,
                config=checkpoint_config,
            )
        progress_tracker.update(round_metrics, len(checkpoints["pilot"]), len(checkpoints["enemies"]))
        return dominance_reached

    try:
        if balanced_rounds is not None:
            phase_number = phase_number_offset
            while len(history) < balanced_rounds:
                if balanced_stop_reached(
                    history,
                    min_balanced_rounds=min_balanced_rounds,
                    balance_patience=balance_patience,
                    dominance_threshold=dominance_threshold,
                    balance_tolerance=balance_tolerance,
                    balance_min_win_rate=balance_min_win_rate,
                ):
                    break
                role = choose_balanced_role(
                    history[-1] if history else None,
                    checkpoints,
                    dominance_threshold=dominance_threshold,
                )
                phase_number += 1
                for phase_iteration in range(1, max_phase_iterations + 1):
                    if len(history) >= balanced_rounds:
                        break
                    dominance_reached = run_generation(role, phase_number, phase_iteration)
                    if dominance_reached or balanced_stop_reached(
                        history,
                        min_balanced_rounds=min_balanced_rounds,
                        balance_patience=balance_patience,
                        dominance_threshold=dominance_threshold,
                        balance_tolerance=balance_tolerance,
                        balance_min_win_rate=balance_min_win_rate,
                    ):
                        break
        elif generations_per_side is not None:
            phase_number = phase_number_offset
            role = "pilot" if len(checkpoints["pilot"]) <= len(checkpoints["enemies"]) else "enemies"
            while len(checkpoints["pilot"]) < generations_per_side or len(checkpoints["enemies"]) < generations_per_side:
                if len(checkpoints[role]) >= generations_per_side:
                    role = "enemies" if role == "pilot" else "pilot"
                    continue
                phase_number += 1
                for phase_iteration in range(1, max_phase_iterations + 1):
                    if len(checkpoints[role]) >= generations_per_side:
                        break
                    dominance_reached = run_generation(role, phase_number, phase_iteration)
                    if dominance_reached:
                        break
                role = "enemies" if role == "pilot" else "pilot"
        else:
            remaining_phase_roles = phase_roles[round_number:] if rounds is not None else phase_roles[phase_number_offset:]
            for phase_index, role in enumerate(remaining_phase_roles, start=phase_number_offset + 1):
                for phase_iteration in range(1, max_phase_iterations + 1):
                    dominance_reached = run_generation(role, phase_index, phase_iteration)
                    if dominance_reached:
                        break
    finally:
        progress_tracker.close()

    if pilot_model is None or enemy_model is None:
        raise RuntimeError("Training requires at least one pilot phase and one enemy phase.")

    return pilot_model, enemy_model, {
        "type": "stable-baselines3-dqn-galagAI-dominance-self-play",
        "rounds": history,
        "latest": history[-1],
        "checkpoints": checkpoints,
        "cycles": cycles,
        "generationsPerSide": generations_per_side,
        "balancedRounds": balanced_rounds,
        "minBalancedRounds": min_balanced_rounds,
        "balanceTolerance": balance_tolerance,
        "balancePatience": balance_patience,
        "balanceMinWinRate": balance_min_win_rate,
        "phaseTimesteps": phase_timesteps,
        "maxPhaseIterations": max_phase_iterations,
        "dominanceThreshold": dominance_threshold,
        "netArch": DQN_NET_ARCH,
        "checkpointDir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "resumedFromCheckpoint": resume,
        "trainWorkers": train_workers,
        "evalWorkers": eval_workers,
        "environment": {
            "name": "HeadlessGalagai",
            "openingEnemyPolicy": "drift-only bootstrap until the first learned enemy checkpoint exists",
            "dropCooldownSeconds": DROP_COOLDOWN_SECONDS,
            "enemyShotCooldownSeconds": ENEMY_SHOT_COOLDOWN_SECONDS,
            "actionDtSeconds": ACTION_DT,
            "antiDropSpam": "invalid drops are ignored and penalized",
        },
    }


def evaluate_current_matchup(
    *,
    seed: int,
    pilot_model: DQN | None,
    enemy_model: DQN | None,
    eval_episodes: int,
    max_steps: int,
    eval_workers: int = 1,
) -> dict[str, object]:
    return evaluate_policy_specs(
        seed=seed,
        pilot_spec=pilot_policy_spec(pilot_model),
        enemy_spec=enemy_policy_spec(enemy_model),
        episodes=eval_episodes,
        max_steps=max_steps,
        workers=eval_workers,
    )


def checkpoint_entry(role: str, model: DQN, version_id: int, metrics: dict[str, object]) -> dict[str, object]:
    if role == "pilot":
        actions = PILOT_ACTIONS
        model_name = f"sb3-dqn-pilot-v{version_id}"
        label = f"Pilot v{version_id}"
    elif role == "enemies":
        actions = ENEMY_ACTIONS
        model_name = f"sb3-dqn-enemies-v{version_id}"
        label = f"Enemies v{version_id}"
    else:
        raise ValueError(f"Unknown checkpoint role {role}.")

    metric_fields = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, str, bool))
    }
    return {
        "id": version_id,
        "label": label,
        "role": role,
        "model": model_name,
        "actions": actions,
        "features": FEATURES,
        "network": export_network(model),
        **metric_fields,
    }


def checkpoint_filename(role: str, version_id: int) -> str:
    prefix = "pilot" if role == "pilot" else "enemies"
    return f"{prefix}-v{version_id:03d}.json"


def manifest_checkpoint(entry: dict[str, object], url: str) -> dict[str, object]:
    return {
        key: value
        for key, value in entry.items()
        if key != "network"
    } | {"url": url}


def run_episode(seed: int, pilot_policy: Policy, enemy_policy: Policy, max_steps: int) -> EpisodeResult:
    game = HeadlessGalagai(seed=seed, max_steps=max_steps)
    observation = game.reset(seed=seed)
    done = False
    info: dict[str, object] = {}
    while not done:
        pilot_action = pilot_policy.act(observation)
        enemy_action = enemy_policy.act(observation)
        observation, _, _, done, info = game.step(pilot_action, enemy_action)
    drop_attempts = max(1, int(info.get("dropAttempts", 0)))
    pilot_fires = int(info.get("pilotFires", 0))
    pilot_hits = int(info.get("pilotHits", 0))
    wave_cleared = int(info.get("wave", 1)) > 1
    return EpisodeResult(
        winner=str(info.get("winner", "none")),
        score=int(info.get("score", 0)),
        wave=int(info.get("wave", 1)),
        steps=game.steps,
        enemy_drop_rate=float(info.get("dropAttempts", 0)) / max(1, game.steps),
        invalid_drop_rate=float(info.get("invalidDrops", 0)) / drop_attempts,
        enemy_fire_rate=float(info.get("enemyFires", 0)) / max(1, game.steps),
        pilot_fire_rate=float(pilot_fires) / max(1, game.steps),
        pilot_hits=pilot_hits,
        pilot_fires=pilot_fires,
        pilot_shot_accuracy=float(pilot_hits) / max(1, pilot_fires),
        wave_cleared=wave_cleared,
        lives_left=int(info.get("lives", 0)),
    )


def run_episode_from_specs(args: tuple[int, PolicySpec, PolicySpec, int]) -> EpisodeResult:
    seed, pilot_spec, enemy_spec, max_steps = args
    return run_episode(seed, policy_from_spec(pilot_spec), policy_from_spec(enemy_spec), max_steps)


def evaluate_policy_specs(
    *,
    seed: int,
    pilot_spec: PolicySpec,
    enemy_spec: PolicySpec,
    episodes: int,
    max_steps: int,
    workers: int,
) -> dict[str, object]:
    if workers > 1 and episodes > 1:
        tasks = [(seed + episode, pilot_spec, enemy_spec, max_steps) for episode in range(episodes)]
        with ProcessPoolExecutor(max_workers=min(workers, episodes)) as executor:
            results = list(executor.map(run_episode_from_specs, tasks))
    else:
        pilot_policy = policy_from_spec(pilot_spec)
        enemy_policy = policy_from_spec(enemy_spec)
        results = [
            run_episode(seed + episode, pilot_policy, enemy_policy, max_steps)
            for episode in range(episodes)
        ]
    return summarize_episode_results(results, episodes)


def evaluate(seed: int, pilot_policy: Policy, enemy_policy: Policy, episodes: int, max_steps: int) -> dict[str, object]:
    results: list[EpisodeResult] = []
    for episode in range(episodes):
        results.append(run_episode(seed + episode, pilot_policy, enemy_policy, max_steps))
    return summarize_episode_results(results, episodes)


def summarize_episode_results(results: list[EpisodeResult], episodes: int) -> dict[str, object]:
    pilot_wins = sum(1 for result in results if result.winner == "pilot")
    enemy_wins = sum(1 for result in results if result.winner == "enemies")
    wave_clear_steps = [result.steps for result in results if result.wave_cleared]
    pilot_hits = sum(result.pilot_hits for result in results)
    pilot_fires = sum(result.pilot_fires for result in results)
    return {
        "pilotWinRate": round(pilot_wins / max(1, episodes), 4),
        "enemyWinRate": round(enemy_wins / max(1, episodes), 4),
        "averageScore": round(float(np.mean([result.score for result in results])), 2),
        "averageWave": round(float(np.mean([result.wave for result in results])), 2),
        "averageSteps": round(float(np.mean([result.steps for result in results])), 2),
        "waveClearRate": round(len(wave_clear_steps) / max(1, episodes), 4),
        "averageClearSteps": round(float(np.mean(wave_clear_steps)), 2) if wave_clear_steps else 0.0,
        "enemyDropRate": round(float(np.mean([result.enemy_drop_rate for result in results])), 4),
        "invalidDropRate": round(float(np.mean([result.invalid_drop_rate for result in results])), 4),
        "enemyFireRate": round(float(np.mean([result.enemy_fire_rate for result in results])), 4),
        "pilotFireRate": round(float(np.mean([result.pilot_fire_rate for result in results])), 4),
        "pilotShotAccuracy": round(pilot_hits / max(1, pilot_fires), 4),
        "pilotHits": pilot_hits,
        "pilotFires": pilot_fires,
    }


def export_network(model: DQN) -> dict[str, object]:
    layers = []
    for module in model.policy.q_net.modules():
        if isinstance(module, nn.Linear):
            weights = module.weight.detach().cpu().numpy().T
            biases = module.bias.detach().cpu().numpy()
            layers.append(
                {
                    "weights": [[round(float(value), 6) for value in row] for row in weights.tolist()],
                    "biases": [round(float(value), 6) for value in biases.tolist()],
                }
            )
    return {"activation": "relu", "layers": layers}


def write_model(path: Path, pilot_model: DQN, enemy_model: DQN, self_play: dict[str, object]) -> None:
    latest = self_play["latest"]
    checkpoints = self_play.get("checkpoints", {})
    pilot_versions = list(checkpoints.get("pilot", [])) if isinstance(checkpoints, dict) else []
    enemy_versions = list(checkpoints.get("enemies", [])) if isinstance(checkpoints, dict) else []
    if not pilot_versions:
        pilot_versions = [checkpoint_entry("pilot", pilot_model, 1, latest)]
    if not enemy_versions:
        enemy_versions = [checkpoint_entry("enemies", enemy_model, 1, latest)]

    self_play_metrics = {
        key: copy.deepcopy(value)
        for key, value in self_play.items()
        if key != "checkpoints"
    }
    self_play_metrics["checkpointCounts"] = {
        "pilot": len(pilot_versions),
        "enemies": len(enemy_versions),
    }
    model_dir = path.parent / MODEL_FILE_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in model_dir.glob("*.json"):
        stale_file.unlink()

    pilot_manifest_versions = []
    for entry in pilot_versions:
        filename = checkpoint_filename("pilot", int(entry["id"]))
        (model_dir / filename).write_text(json.dumps(entry, separators=(",", ":")) + "\n", encoding="utf-8")
        pilot_manifest_versions.append(manifest_checkpoint(entry, f"{MODEL_FILE_DIR}/{filename}"))

    enemy_manifest_versions = []
    for entry in enemy_versions:
        filename = checkpoint_filename("enemies", int(entry["id"]))
        (model_dir / filename).write_text(json.dumps(entry, separators=(",", ":")) + "\n", encoding="utf-8")
        enemy_manifest_versions.append(manifest_checkpoint(entry, f"{MODEL_FILE_DIR}/{filename}"))

    payload = {
        "model": "sb3-dqn-pilot",
        "version": MODEL_SCHEMA_VERSION,
        "algorithm": "stable-baselines3-dqn",
        "actions": PILOT_ACTIONS,
        "features": FEATURES,
        "networkRef": pilot_manifest_versions[-1]["url"],
        "versions": {
            "pilot": pilot_manifest_versions,
            "enemies": enemy_manifest_versions,
        },
        "enemies": {
            "model": "sb3-dqn-enemies",
            "actions": ENEMY_ACTIONS,
            "features": FEATURES,
            "networkRef": enemy_manifest_versions[-1]["url"],
            "constraints": {
                "dropCooldownSeconds": DROP_COOLDOWN_SECONDS,
                "shotCooldownSeconds": ENEMY_SHOT_COOLDOWN_SECONDS,
                "invalidDropPenalty": INVALID_DROP_PENALTY,
            },
        },
        "metrics": {
            "rlAlgorithm": "stable-baselines3-dqn",
            "evalAccuracy": float(latest["pilotWinRate"]),
            "enemyWinRate": float(latest["enemyWinRate"]),
            "enemyDropRate": float(latest["enemyDropRate"]),
            "invalidDropRate": float(latest["invalidDropRate"]),
            "enemyFireRate": float(latest["enemyFireRate"]),
            "pilotShotAccuracy": float(latest.get("pilotShotAccuracy", 0.0)),
            "waveClearRate": float(latest.get("waveClearRate", 0.0)),
            "averageClearSteps": float(latest.get("averageClearSteps", 0.0)),
            "selfPlay": self_play_metrics,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def artifact_summary(path: Path) -> dict[str, int]:
    model_dir = path.parent / MODEL_FILE_DIR
    checkpoint_files = list(model_dir.glob("*.json")) if model_dir.exists() else []
    return {
        "manifestBytes": path.stat().st_size if path.exists() else 0,
        "checkpointFiles": len(checkpoint_files),
        "checkpointBytes": sum(file.stat().st_size for file in checkpoint_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train static GalagAI pilot/enemy SB3 DQN policies.")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--phase-timesteps", type=int, default=2200)
    parser.add_argument("--dominance-threshold", type=float, default=0.6)
    parser.add_argument("--max-phase-iterations", type=int, default=3)
    parser.add_argument("--generations-per-side", type=int, default=None)
    parser.add_argument("--balanced-rounds", type=int, default=None, help="Maximum adaptive dominance-balanced generations.")
    parser.add_argument("--min-balanced-rounds", type=int, default=6)
    parser.add_argument("--balance-tolerance", type=float, default=0.18)
    parser.add_argument("--balance-patience", type=int, default=3)
    parser.add_argument("--balance-min-win-rate", type=float, default=0.25)
    parser.add_argument("--rounds", type=int, default=None, help="Deprecated fixed alternation phase count.")
    parser.add_argument("--timesteps-per-round", type=int, default=None, help="Deprecated alias for --phase-timesteps.")
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--out", type=Path, default=Path("js/galagai-model.json"))
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint-dir instead of starting fresh.")
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable per-generation trainer checkpoints.")
    parser.add_argument("--train-workers", type=int, default=1, help="Parallel headless envs for SB3 rollout collection.")
    parser.add_argument("--eval-workers", type=int, default=1, help="Parallel processes for dominance evaluation episodes.")
    args = parser.parse_args()

    pilot_model, enemy_model, self_play = train_self_play(
        seed=args.seed,
        cycles=args.cycles,
        phase_timesteps=args.phase_timesteps,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        dominance_threshold=args.dominance_threshold,
        max_phase_iterations=args.max_phase_iterations,
        generations_per_side=args.generations_per_side,
        balanced_rounds=args.balanced_rounds,
        min_balanced_rounds=args.min_balanced_rounds,
        balance_tolerance=args.balance_tolerance,
        balance_patience=args.balance_patience,
        balance_min_win_rate=args.balance_min_win_rate,
        rounds=args.rounds,
        timesteps_per_round=args.timesteps_per_round,
        progress=not args.no_progress,
        checkpoint_dir=None if args.no_checkpoints else args.checkpoint_dir,
        resume=args.resume,
        train_workers=args.train_workers,
        eval_workers=args.eval_workers,
    )
    write_model(args.out, pilot_model, enemy_model, self_play)
    summary = artifact_summary(args.out)
    print(
        json.dumps(
            {
                "model": str(args.out),
                "algorithm": "stable-baselines3-dqn",
                "cycles": self_play["cycles"],
                "generationsPerSide": self_play["generationsPerSide"],
                "balancedRounds": self_play["balancedRounds"],
                "minBalancedRounds": self_play["minBalancedRounds"],
                "balanceTolerance": self_play["balanceTolerance"],
                "balancePatience": self_play["balancePatience"],
                "balanceMinWinRate": self_play["balanceMinWinRate"],
                "phaseTimesteps": self_play["phaseTimesteps"],
                "dominanceThreshold": self_play["dominanceThreshold"],
                "maxPhaseIterations": self_play["maxPhaseIterations"],
                "netArch": self_play["netArch"],
                "checkpointDir": self_play["checkpointDir"],
                "resumedFromCheckpoint": self_play["resumedFromCheckpoint"],
                "trainWorkers": self_play["trainWorkers"],
                "evalWorkers": self_play["evalWorkers"],
                "roundsCompleted": len(self_play["rounds"]),
                "checkpointCounts": {
                    "pilot": len(self_play["checkpoints"]["pilot"]),
                    "enemies": len(self_play["checkpoints"]["enemies"]),
                },
                "evalEpisodes": args.eval_episodes,
                "artifact": summary,
                "selfPlayLatest": self_play["latest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
