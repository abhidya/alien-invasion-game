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


PILOT_ACTIONS = ["left", "right", "fire", "stay", "up", "down"]
ENEMY_ACTIONS = [
    "hold",
    "left",
    "right",
    "down",
    "left_fire",
    "right_fire",
    "down_fire",
    "fire",
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
    "danger_shot_dx",
    "danger_shot_y",
    "danger_shot_lane",
    "pilot_bullet_dx",
    "pilot_bullet_y",
    "bee_count",
    "gunship_count",
    "boss_count",
    "ship_y",
    "target_dy",
    "enemy_ship_dx",
    "enemy_ship_y",
    "enemy_ship_role",
    "enemy_ship_fire_ready",
    "enemy_ship_bullet_lane",
    "enemy_ship_bottom",
]

CANVAS_WIDTH = 960.0
CANVAS_HEIGHT = 560.0
SHIP_WIDTH = 64.0
SHIP_HEIGHT = 48.0
SHIP_Y = CANVAS_HEIGHT - 72.0
SHIP_SPEED = 470.0
SHIP_VERTICAL_SPEED = 330.0
SHIP_MIN_Y = CANVAS_HEIGHT - 170.0
SHIP_MAX_Y = CANVAS_HEIGHT - 56.0
BULLET_SPEED = 620.0
ENEMY_SHOT_SPEED = 230.0
ALIEN_WIDTH = 48.0
ALIEN_HEIGHT = 34.0
FLEET_DROP = 18.0
MAX_ALIENS_NORMALIZER = 45.0
ACTION_DT = 0.12
DROP_COOLDOWN_SECONDS = 1.08
ENEMY_SHOT_COOLDOWN_SECONDS = 0.0
ENEMY_SHIP_DOWN_COOLDOWN_SECONDS = 0.45
ENEMY_SHIP_SHOT_COOLDOWN_SECONDS = 0.65
ENEMY_SHIP_CONTROL_STEP_X = 32.0
ENEMY_SHIP_CONTROL_STEP_Y = FLEET_DROP * 0.70
MAX_ENEMY_SHOTS_PER_STEP = 4
INVALID_DROP_PENALTY = 0.90
MODEL_SCHEMA_VERSION = 14
DQN_NET_ARCH = [64, 64]
MODEL_FILE_DIR = "galagai-models"
DEFAULT_CHECKPOINT_DIR = Path(".training-checkpoints/galagai-balanced-v14")
RETENTION_LATEST_DEFAULT = 12


@dataclass(frozen=True)
class CheckpointRetention:
    mode: str = "all"
    keep_latest: int = RETENTION_LATEST_DEFAULT

    def __post_init__(self) -> None:
        if self.mode not in {"all", "tiered"}:
            raise ValueError("checkpoint retention mode must be 'all' or 'tiered'.")
        if self.keep_latest < 1:
            raise ValueError("keep_latest must be at least 1.")

    def to_json(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "keepLatest": self.keep_latest,
            "tieredRule": "keep generation 1, latest N, every 2 through 100, every 10 through 1000, every 100 after",
        }


@dataclass
class Actor:
    x: float
    y: float
    width: float
    height: float
    alive: bool = True
    role: str = "bee"
    shot_cooldown: float = 0.0
    down_cooldown: float = 0.0

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

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


@dataclass(frozen=True)
class CandidateResult:
    spawn_index: int
    metrics: dict[str, object]
    model_path: str
    replay_path: str


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
        danger_dx = float(observation[11]) if len(observation) > 13 else threat_dx
        danger_y = float(observation[12]) if len(observation) > 13 else threat_y
        danger_lane = float(observation[13]) if len(observation) > 13 else 0.0
        target_dy = float(observation[20]) if len(observation) > 20 else 0.0
        if danger_y > 0.44 and danger_lane > 0.20:
            return 0 if danger_dx >= 0 else 1
        if threat_y > 0.74 and abs(threat_dx) < 0.18:
            return 0 if threat_dx >= 0 else 1
        if bullet_ready and abs(target_dx) < 0.10:
            return 2
        if abs(target_dx) < 0.16 and target_dy < -0.24:
            return 4
        if abs(target_dx) < 0.16 and target_dy > 0.18:
            return 5
        if target_dx < -0.08:
            return 0
        if target_dx > 0.08:
            return 1
        return 3


class HeuristicEnemyPolicy:
    def act(self, observation: np.ndarray) -> int:
        target_dx = float(observation[0])
        drop_ready = observation[7] > 0.5
        enemy_dx = float(observation[21]) if len(observation) > 21 else target_dx
        enemy_y = float(observation[22]) if len(observation) > 22 else float(observation[9])
        fire_ready = observation[24] > 0.5 if len(observation) > 24 else False
        bullet_lane = float(observation[25]) if len(observation) > 25 else 0.0
        if bullet_lane > 0.45:
            return 1 if enemy_dx >= target_dx else 2
        if fire_ready and abs(enemy_dx) < 0.18:
            return 7
        if fire_ready and abs(enemy_dx) < 0.34:
            return 4 if target_dx < 0 else 5
        if drop_ready and enemy_y < 0.58 and abs(enemy_dx) > 0.22:
            return 3
        return 1 if target_dx < 0 else 2


class OpeningEnemyPolicy:
    def act(self, observation: np.ndarray) -> int:
        target_dx = float(observation[0])
        drop_ready = observation[7] > 0.5
        enemy_dx = float(observation[21]) if len(observation) > 21 else target_dx
        enemy_y = float(observation[22]) if len(observation) > 22 else float(observation[9])
        fire_ready = observation[24] > 0.5 if len(observation) > 24 else False
        if fire_ready and abs(enemy_dx) < 0.24:
            return 7
        if fire_ready:
            return 4 if target_dx < 0 else 5
        if drop_ready and enemy_y < 0.48 and abs(enemy_dx) > 0.30:
            return 3
        return 1 if target_dx < 0 else 2


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

    def __init__(self, seed: int = 0, max_steps: int = 520, max_start_wave: int = 1):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.max_start_wave = max(1, int(max_start_wave))
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.score = 0
        self.start_wave = (
            int(self.rng.integers(1, self.max_start_wave + 1))
            if self.max_start_wave > 1
            else 1
        )
        self.wave = self.start_wave
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
        self.enemy_control_cursor = 0
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

    def features(self, controlled_alien: Actor | None = None) -> np.ndarray:
        ship_center = self.ship.center_x
        live_aliens = [alien for alien in self.aliens if alien.alive]
        target = min(live_aliens, key=lambda alien: abs(alien.center_x - ship_center), default=None)
        controlled = controlled_alien if controlled_alien is not None and controlled_alien.alive else target
        target_dx = self._relative_x(target.center_x - ship_center) if target else 0.0
        target_dy = self._clamp(
            ((target.center_y - self.ship.center_y) / CANVAS_HEIGHT) if target else 0.0,
            -1.0,
            1.0,
        )

        closest_shot = min(self.enemy_shots, key=lambda shot: abs(shot.center_x - ship_center), default=None)
        danger_shot = self.dangerous_enemy_shot()
        bullet_threat = self.dangerous_pilot_bullet(live_aliens)
        threat_dx = self._relative_x(closest_shot.center_x - ship_center) if closest_shot else 1.0
        threat_y = self._clamp01(closest_shot.y / CANVAS_HEIGHT) if closest_shot else 0.0
        danger_dx = self._relative_x(danger_shot.center_x - ship_center) if danger_shot else 1.0
        danger_y = self._clamp01(danger_shot.y / CANVAS_HEIGHT) if danger_shot else 0.0
        danger_lane = self.shot_lane_overlap(danger_shot, self.ship) if danger_shot else 0.0
        if bullet_threat is not None:
            pilot_bullet, threatened_alien = bullet_threat
            pilot_bullet_dx = self._relative_x(pilot_bullet.center_x - threatened_alien.center_x)
            pilot_bullet_y = self._clamp(
                (pilot_bullet.y - threatened_alien.bottom) / CANVAS_HEIGHT,
                -1.0,
                1.0,
            )
        else:
            pilot_bullet_dx = 0.0
            pilot_bullet_y = 0.0
        fleet_y = self._clamp01(max((alien.y for alien in live_aliens), default=0.0) / CANVAS_HEIGHT)
        bee_count = sum(1 for alien in live_aliens if alien.role == "bee")
        gunship_count = sum(1 for alien in live_aliens if alien.role in {"butterfly", "boss"})
        boss_count = sum(1 for alien in live_aliens if alien.role == "boss")
        enemy_ship_dx = self._relative_x(controlled.center_x - ship_center) if controlled else 0.0
        enemy_ship_y = self._clamp01(controlled.y / CANVAS_HEIGHT) if controlled else 0.0
        enemy_ship_role = self.enemy_role_value(controlled.role) if controlled else 0.0
        enemy_ship_fire_ready = 1.0 if controlled is not None and self.can_enemy_fire(controlled) else 0.0
        enemy_ship_bullet_lane = self.pilot_bullet_lane_for(controlled) if controlled else 0.0
        enemy_ship_bottom = self._clamp01(controlled.bottom / CANVAS_HEIGHT) if controlled else 0.0

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
                danger_dx,
                danger_y,
                danger_lane,
                pilot_bullet_dx,
                pilot_bullet_y,
                self._clamp01(bee_count / MAX_ALIENS_NORMALIZER),
                self._clamp01(gunship_count / MAX_ALIENS_NORMALIZER),
                self._clamp01(boss_count / MAX_ALIENS_NORMALIZER),
                self._clamp01(self.ship.y / CANVAS_HEIGHT),
                target_dy,
                enemy_ship_dx,
                enemy_ship_y,
                enemy_ship_role,
                enemy_ship_fire_ready,
                enemy_ship_bullet_lane,
                enemy_ship_bottom,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def enemy_role_value(role: str) -> float:
        if role == "boss":
            return 1.0
        if role == "butterfly":
            return 0.5
        return 0.0

    def selected_enemy(self) -> Actor | None:
        live_aliens = [alien for alien in self.aliens if alien.alive]
        if not live_aliens:
            return None
        return live_aliens[self.enemy_control_cursor % len(live_aliens)]

    def enemy_control_observation(self) -> np.ndarray:
        return self.features(self.selected_enemy())

    def step(
        self,
        pilot_action: int,
        enemy_action: int | list[tuple[Actor, int]],
    ) -> tuple[np.ndarray, float, float, bool, dict[str, object]]:
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
        for alien in self.aliens:
            alien.shot_cooldown = max(0.0, alien.shot_cooldown - ACTION_DT)
            alien.down_cooldown = max(0.0, alien.down_cooldown - ACTION_DT)

        action_features = self.features()
        target_dx = float(action_features[0])
        fleet_y = float(action_features[9])
        pilot_bullet_dx = float(action_features[14])
        pilot_bullet_y = float(action_features[15])
        target_dy = float(action_features[20])
        enemy_actions = self.enemy_action_targets(enemy_action)
        pilot_aligned_action = self.is_pilot_aligned_action(pilot_action, target_dx, target_dy)
        pilot_aligned_fire = pilot_action == 2 and self.fire_cooldown <= 0 and abs(target_dx) < 0.16
        pilot_bad_fire = pilot_action == 2 and self.fire_cooldown <= 0 and abs(target_dx) > 0.30
        enemy_tactical_action = any(
            self.is_enemy_tactical_action(action, target_dx, fleet_y, pilot_bullet_dx, pilot_bullet_y)
            for _, action in enemy_actions
        )

        if pilot_action == 0:
            self.ship.x -= SHIP_SPEED * ACTION_DT
        elif pilot_action == 1:
            self.ship.x += SHIP_SPEED * ACTION_DT
        elif pilot_action == 2:
            pilot_fired, pilot_missed = self.fire_pilot_bullet()
        elif pilot_action == 4:
            self.ship.y -= SHIP_VERTICAL_SPEED * ACTION_DT
        elif pilot_action == 5:
            self.ship.y += SHIP_VERTICAL_SPEED * ACTION_DT
        elif pilot_action != 3:
            raise ValueError(f"Unknown pilot action {pilot_action}.")
        self.wrap_ship_horizontal()
        self.ship.y = self._clamp(self.ship.y, SHIP_MIN_Y, SHIP_MAX_Y)

        enemy_events = self.apply_enemy_actions(enemy_actions)
        enemy_fired = enemy_events["enemy_fires"] > 0
        invalid_fire = enemy_events["invalid_fires"] > 0
        valid_drop = enemy_events["valid_downs"] > 0
        invalid_drop = enemy_events["invalid_downs"] > 0

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
            "startWave": self.start_wave,
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

    def enemy_action_targets(self, enemy_action: int | list[tuple[Actor, int]]) -> list[tuple[Actor, int]]:
        if isinstance(enemy_action, list):
            return [(alien, int(action)) for alien, action in enemy_action if alien.alive]
        selected = self.selected_enemy()
        self.enemy_control_cursor += 1
        return [(selected, int(enemy_action))] if selected is not None else []

    def enemy_policy_actions(self, policy: Policy) -> list[tuple[Actor, int]]:
        return [
            (alien, policy.act(self.features(alien)))
            for alien in self.aliens
            if alien.alive
        ]

    def apply_enemy_actions(self, actions: list[tuple[Actor, int]]) -> dict[str, int]:
        events = {"enemy_fires": 0, "invalid_fires": 0, "valid_downs": 0, "invalid_downs": 0}
        shots_this_step = 0
        for alien, action_index in actions:
            if not alien.alive:
                continue
            if action_index < 0 or action_index >= len(ENEMY_ACTIONS):
                raise ValueError(f"Unknown enemy action {action_index}.")
            action = ENEMY_ACTIONS[action_index]
            if action in {"left", "left_fire"}:
                alien.x -= ENEMY_SHIP_CONTROL_STEP_X
                self.wrap_alien_horizontal(alien)
            elif action in {"right", "right_fire"}:
                alien.x += ENEMY_SHIP_CONTROL_STEP_X
                self.wrap_alien_horizontal(alien)

            if action in {"down", "down_fire"}:
                self.drop_attempts += 1
                if alien.down_cooldown <= 0.0 and alien.bottom < CANVAS_HEIGHT:
                    alien.y += ENEMY_SHIP_CONTROL_STEP_Y
                    alien.down_cooldown = ENEMY_SHIP_DOWN_COOLDOWN_SECONDS
                    events["valid_downs"] += 1
                else:
                    self.invalid_drops += 1
                    events["invalid_downs"] += 1

            if "fire" in action:
                if shots_this_step < MAX_ENEMY_SHOTS_PER_STEP and self.fire_enemy_shot_from(alien):
                    shots_this_step += 1
                    events["enemy_fires"] += 1
                else:
                    events["invalid_fires"] += 1
        return events

    def fire_pilot_bullet(self) -> tuple[bool, bool]:
        if self.fire_cooldown > 0:
            return False, True
        self.bullets.append(Shot(self.ship.center_x - 3.0, self.ship.y - 14.0, 6.0, 18.0, BULLET_SPEED))
        self.fire_cooldown = 0.17
        self.pilot_fires += 1
        return True, False

    def can_enemy_fire(self, alien: Actor) -> bool:
        return alien.alive and alien.role in {"butterfly", "boss"} and alien.shot_cooldown <= 0.0

    def fire_enemy_shot_from(self, alien: Actor) -> bool:
        if not self.can_enemy_fire(alien):
            return False
        self.enemy_shots.append(Shot(alien.center_x - 3.0, alien.bottom, 6.0, 16.0, ENEMY_SHOT_SPEED + self.wave * 20.0))
        alien.shot_cooldown = ENEMY_SHIP_SHOT_COOLDOWN_SECONDS
        self.enemy_fires += 1
        return True

    def fire_enemy_shot(self, roles: tuple[str, ...] | None = None) -> bool:
        live_aliens = [
            alien
            for alien in self.aliens
            if alien.alive and (roles is None or alien.role in roles)
        ]
        if not live_aliens:
            return False
        alien = min(live_aliens, key=lambda item: abs(item.center_x - self.ship.center_x))
        return self.fire_enemy_shot_from(alien)

    def dangerous_enemy_shot(self) -> Shot | None:
        if not self.enemy_shots:
            return None
        ship_center = self.ship.center_x
        return max(
            self.enemy_shots,
            key=lambda shot: self.shot_lane_overlap(shot, self.ship) * 3.0
            + self._clamp01(shot.y / CANVAS_HEIGHT)
            - self._clamp01(max(0.0, self.ship.y - shot.y) / CANVAS_HEIGHT) * 0.35,
        )

    def dangerous_pilot_bullet(self, live_aliens: list[Actor] | None = None) -> tuple[Shot, Actor] | None:
        live_aliens = live_aliens if live_aliens is not None else [alien for alien in self.aliens if alien.alive]
        if not live_aliens or not self.bullets:
            return None
        best: tuple[float, Shot, Actor] | None = None
        for bullet in self.bullets:
            for alien in live_aliens:
                if bullet.y + bullet.height < alien.y - 8.0:
                    continue
                lane_score = self.shot_lane_overlap(bullet, alien)
                vertical_gap = abs(bullet.y - alien.bottom)
                vertical_score = 1.0 - self._clamp01(vertical_gap / CANVAS_HEIGHT)
                score = lane_score * 3.0 + vertical_score
                if best is None or score > best[0]:
                    best = (score, bullet, alien)
        if best is None:
            return None
        return best[1], best[2]

    def pilot_bullet_lane_for(self, alien: Actor) -> float:
        return max((self.shot_lane_overlap(bullet, alien) for bullet in self.bullets), default=0.0)

    @staticmethod
    def shot_lane_overlap(shot: Shot, actor: Actor) -> float:
        lane_width = max(actor.width * 0.85, actor.width / 2.0 + shot.width / 2.0)
        distance = abs(shot.center_x - actor.center_x)
        return max(0.0, min(1.0, 1.0 - distance / lane_width))

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
        for alien in live_aliens:
            alien.x += self.fleet_direction * self.fleet_speed * ACTION_DT
            self.wrap_alien_horizontal(alien)
            if alien.alive and self.intersects(alien, self.ship) and self.invulnerability <= 0:
                alien.alive = False
                self.lose_life()
            elif alien.bottom >= CANVAS_HEIGHT:
                alien.alive = False

    @staticmethod
    def wrap_alien_horizontal(alien: Actor) -> None:
        if alien.x + alien.width < 0.0:
            alien.x = CANVAS_WIDTH
        elif alien.x > CANVAS_WIDTH:
            alien.x = -alien.width

    def wrap_ship_horizontal(self) -> None:
        if self.ship.x + self.ship.width < 0.0:
            self.ship.x = CANVAS_WIDTH
        elif self.ship.x > CANVAS_WIDTH:
            self.ship.x = -self.ship.width

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
        self.ship.y = SHIP_Y

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
        if self.live_alien_count == 0 or self.wave > self.start_wave:
            return "pilot"
        return "timeout"

    @staticmethod
    def is_pilot_aligned_action(action: int, target_dx: float, target_dy: float = 0.0) -> bool:
        if action == 0:
            return target_dx < -0.06
        if action == 1:
            return target_dx > 0.06
        if action == 2:
            return abs(target_dx) < 0.16
        if action == 3:
            return abs(target_dx) < 0.05
        if action == 4:
            return target_dy < -0.08
        if action == 5:
            return target_dy > 0.08
        return False

    @staticmethod
    def is_enemy_tactical_action(
        action: int,
        target_dx: float,
        fleet_y: float,
        pilot_bullet_dx: float = 0.0,
        pilot_bullet_y: float = 0.0,
    ) -> bool:
        if pilot_bullet_y > -0.05 and abs(pilot_bullet_dx) < 0.16 and action in {1, 2, 3}:
            return True
        if action == 1:
            return target_dx < -0.06
        if action == 2:
            return target_dx > 0.06
        if action == 3:
            return fleet_y < 0.58
        if action == 4:
            return target_dx < -0.06 or abs(target_dx) < 0.20
        if action == 5:
            return target_dx > 0.06 or abs(target_dx) < 0.20
        if action == 6:
            return fleet_y < 0.58 or abs(target_dx) < 0.24
        if action == 7:
            return abs(target_dx) < 0.24
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

    def __init__(self, opponent: Policy, seed: int = 0, max_steps: int = 520, max_start_wave: int = 1):
        super().__init__()
        self.opponent = opponent
        self.game = HeadlessGalagai(seed=seed, max_steps=max_steps, max_start_wave=max_start_wave)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(len(FEATURES),), dtype=np.float32)
        self.action_space = spaces.Discrete(len(PILOT_ACTIONS))
        self.seed_value = seed

    def reset(self, *, seed: int | None = None, options=None):
        self.seed_value = self.seed_value + 1 if seed is None else seed
        return self.game.reset(seed=self.seed_value), {}

    def step(self, action: int):
        observation = self.game.features()
        enemy_actions = self.game.enemy_policy_actions(self.opponent)
        next_observation, pilot_reward, _, done, info = self.game.step(int(action), enemy_actions)
        terminated = done and str(info.get("winner", "none")) in {"pilot", "enemies"}
        truncated = done and not terminated
        return next_observation, pilot_reward, terminated, truncated, serializable_info(info)


class EnemyTrainingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent: Policy, seed: int = 0, max_steps: int = 520, max_start_wave: int = 1):
        super().__init__()
        self.opponent = opponent
        self.game = HeadlessGalagai(seed=seed, max_steps=max_steps, max_start_wave=max_start_wave)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(len(FEATURES),), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ENEMY_ACTIONS))
        self.seed_value = seed

    def reset(self, *, seed: int | None = None, options=None):
        self.seed_value = self.seed_value + 1 if seed is None else seed
        self.game.reset(seed=self.seed_value)
        return self.game.enemy_control_observation(), {}

    def step(self, action: int):
        pilot_action = self.opponent.act(self.game.features())
        next_observation, _, enemy_reward, done, info = self.game.step(pilot_action, int(action))
        terminated = done and str(info.get("winner", "none")) in {"pilot", "enemies"}
        truncated = done and not terminated
        return self.game.enemy_control_observation(), enemy_reward, terminated, truncated, serializable_info(info)


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


def make_role_env(role: str, opponent_spec: PolicySpec, seed: int, max_steps: int, max_start_wave: int = 1) -> gym.Env:
    opponent = policy_from_spec(opponent_spec)
    if role == "pilot":
        return Monitor(PilotTrainingEnv(opponent, seed=seed, max_steps=max_steps, max_start_wave=max_start_wave))
    if role == "enemies":
        return Monitor(EnemyTrainingEnv(opponent, seed=seed, max_steps=max_steps, max_start_wave=max_start_wave))
    raise ValueError(f"Unknown training role {role}.")


def make_training_env(
    role: str,
    opponent_spec: PolicySpec,
    *,
    seed: int,
    max_steps: int,
    workers: int,
    max_start_wave: int = 1,
) -> Any:
    worker_count = max(1, int(workers))
    if worker_count == 1:
        return make_role_env(role, opponent_spec, seed=seed, max_steps=max_steps, max_start_wave=max_start_wave)

    def make_env(worker_index: int):
        def _init() -> gym.Env:
            return make_role_env(
                role,
                opponent_spec,
                seed=seed + worker_index * 10_000,
                max_steps=max_steps,
                max_start_wave=max_start_wave,
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


def train_role_model(
    *,
    role: str,
    model: DQN | None,
    opponent_spec: PolicySpec,
    seed: int,
    phase_timesteps: int,
    max_steps: int,
    train_workers: int,
    curriculum_waves: int,
) -> DQN:
    env = make_training_env(
        role,
        opponent_spec,
        seed=seed,
        max_steps=max_steps,
        workers=train_workers,
        max_start_wave=curriculum_waves,
    )
    try:
        if model is None:
            model = make_dqn(env, seed=seed, learning_rate=8e-4)
        else:
            model.set_env(env)
            set_random_seed = getattr(model, "set_random_seed", None)
            if callable(set_random_seed):
                set_random_seed(seed)
        model.learn(total_timesteps=phase_timesteps, reset_num_timesteps=False, progress_bar=False)
        return model
    finally:
        close_training_env(env)


def train_candidate_from_files(args: tuple[object, ...]) -> CandidateResult:
    (
        role,
        spawn_index,
        base_model_path,
        base_replay_path,
        opponent_spec,
        seed,
        eval_seed,
        phase_timesteps,
        max_steps,
        train_workers,
        eval_episodes,
        eval_workers,
        curriculum_waves,
        output_dir,
    ) = args
    role = str(role)
    spawn_index = int(spawn_index)
    output_path = Path(str(output_dir))
    base_model = None
    if base_model_path is not None:
        base_model = DQN.load(str(base_model_path))
        if base_replay_path is not None and Path(str(base_replay_path)).exists():
            base_model.load_replay_buffer(str(base_replay_path))

    trained_model = train_role_model(
        role=role,
        model=base_model,
        opponent_spec=opponent_spec,  # type: ignore[arg-type]
        seed=int(seed),
        phase_timesteps=int(phase_timesteps),
        max_steps=int(max_steps),
        train_workers=int(train_workers),
        curriculum_waves=int(curriculum_waves),
    )
    candidate_spec = {
        "kind": "network",
        "role": role,
        "network": export_network(trained_model),
    }
    if role == "pilot":
        metrics = evaluate_policy_specs(
            seed=int(eval_seed),
            pilot_spec=candidate_spec,
            enemy_spec=opponent_spec,  # type: ignore[arg-type]
            episodes=int(eval_episodes),
            max_steps=int(max_steps),
            workers=int(eval_workers),
            curriculum_waves=int(curriculum_waves),
        )
    else:
        metrics = evaluate_policy_specs(
            seed=int(eval_seed),
            pilot_spec=opponent_spec,  # type: ignore[arg-type]
            enemy_spec=candidate_spec,
            episodes=int(eval_episodes),
            max_steps=int(max_steps),
            workers=int(eval_workers),
            curriculum_waves=int(curriculum_waves),
        )

    model_path = output_path / f"{role}-candidate-{spawn_index}.zip"
    replay_path = output_path / f"{role}-candidate-{spawn_index}-replay.pkl"
    trained_model.save(model_path)
    trained_model.save_replay_buffer(replay_path)
    return CandidateResult(
        spawn_index=spawn_index,
        metrics=metrics,
        model_path=str(model_path),
        replay_path=str(replay_path),
    )


def save_candidate_base_model(model: DQN | None, directory: Path, role: str) -> tuple[str | None, str | None]:
    if model is None:
        return None, None
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f"{role}-base.zip"
    replay_path = directory / f"{role}-base-replay.pkl"
    model.save(model_path)
    model.save_replay_buffer(replay_path)
    return str(model_path), str(replay_path)


def load_candidate_model(result: CandidateResult) -> DQN:
    model = DQN.load(result.model_path)
    replay_path = Path(result.replay_path)
    if replay_path.exists():
        model.load_replay_buffer(replay_path)
    return model


def candidate_score(result: CandidateResult, role: str) -> tuple[float, ...]:
    metrics = result.metrics
    if role == "pilot":
        return (
            float(metrics.get("pilotWinRate", 0.0)),
            float(metrics.get("waveClearRate", 0.0)),
            float(metrics.get("pilotShotAccuracy", 0.0)),
            float(metrics.get("averageScore", 0.0)) / 10_000.0,
            -float(metrics.get("averageSteps", 0.0)) / 1_000.0,
        )
    return (
        float(metrics.get("enemyWinRate", 0.0)),
        float(metrics.get("enemyFireRate", 0.0)),
        -float(metrics.get("invalidDropRate", 0.0)),
        -float(metrics.get("pilotShotAccuracy", 0.0)),
        -float(metrics.get("averageScore", 0.0)) / 10_000.0,
    )


def best_candidate_result(results: list[CandidateResult], role: str) -> CandidateResult:
    if not results:
        raise RuntimeError("Candidate spawning produced no trained candidates.")
    return max(results, key=lambda result: candidate_score(result, role))


def candidate_metric_summary(results: list[CandidateResult]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for result in results:
        summaries.append(
            {
                "candidate": result.spawn_index,
                **{
                    key: value
                    for key, value in result.metrics.items()
                    if isinstance(value, (int, float, str, bool))
                },
            }
        )
    return summaries


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


def balanced_stop_reached_after_required_rounds(
    history: list[dict[str, object]],
    *,
    completed_generations: int,
    required_new_balanced_rounds: int,
    min_balanced_rounds: int,
    balance_patience: int,
    dominance_threshold: float,
    balance_tolerance: float,
    balance_min_win_rate: float,
) -> bool:
    required_total = completed_generations + max(0, required_new_balanced_rounds)
    if len(history) < required_total:
        return False
    return balanced_stop_reached(
        history,
        min_balanced_rounds=min_balanced_rounds,
        balance_patience=balance_patience,
        dominance_threshold=dominance_threshold,
        balance_tolerance=balance_tolerance,
        balance_min_win_rate=balance_min_win_rate,
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


def role_generation_count(history: list[dict[str, object]], role: str) -> int:
    return sum(1 for entry in history if entry.get("trained") == role)


def generation_counts(history: list[dict[str, object]]) -> dict[str, int]:
    return {
        "pilot": role_generation_count(history, "pilot"),
        "enemies": role_generation_count(history, "enemies"),
    }


def retained_generation_ids(max_generation: int, retention: CheckpointRetention) -> set[int]:
    if max_generation <= 0:
        return set()
    if retention.mode == "all":
        return set(range(1, max_generation + 1))

    latest_start = max(1, max_generation - retention.keep_latest + 1)
    retained = {1, max_generation, *range(latest_start, max_generation + 1)}
    for generation in range(1, max_generation + 1):
        if generation <= 100 and generation % 2 == 0:
            retained.add(generation)
        elif generation <= 1000 and generation % 10 == 0:
            retained.add(generation)
        elif generation % 100 == 0:
            retained.add(generation)
    return retained


def retain_checkpoint_entries(
    entries: list[dict[str, object]],
    retention: CheckpointRetention,
) -> list[dict[str, object]]:
    if retention.mode == "all" or not entries:
        return list(entries)
    max_generation = max(int(entry["id"]) for entry in entries)
    retained_ids = retained_generation_ids(max_generation, retention)
    return [entry for entry in entries if int(entry["id"]) in retained_ids]


def retain_checkpoint_sets(
    checkpoints: dict[str, list[dict[str, object]]],
    retention: CheckpointRetention,
) -> dict[str, list[dict[str, object]]]:
    return {
        "pilot": retain_checkpoint_entries(checkpoints.get("pilot", []), retention),
        "enemies": retain_checkpoint_entries(checkpoints.get("enemies", []), retention),
    }


class TrainingCheckpointStore:
    def __init__(self, directory: Path, retention: CheckpointRetention | None = None):
        self.directory = directory
        self.exports_dir = directory / "exports"
        self.state_path = directory / "state.json"
        self.retention = retention or CheckpointRetention()

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

        retained_checkpoints = retain_checkpoint_sets(checkpoints, self.retention)
        checkpoint_files: dict[str, list[str]] = {"pilot": [], "enemies": []}
        for role in ("pilot", "enemies"):
            entries = retained_checkpoints.get(role, [])
            retained_filenames = set()
            for entry in entries:
                filename = checkpoint_filename(role, int(entry["id"]))
                entry_path = self.exports_dir / filename
                retained_filenames.add(filename)
                checkpoint_files[role].append(filename)
                if not entry_path.exists() or int(entry["id"]) == max(int(item["id"]) for item in entries):
                    self._write_json_atomic(entry_path, entry)
            prefix = "pilot" if role == "pilot" else "enemies"
            for stale_path in self.exports_dir.glob(f"{prefix}-v*.json"):
                if stale_path.name not in retained_filenames:
                    stale_path.unlink()

        state = {
            "schemaVersion": MODEL_SCHEMA_VERSION,
            "algorithm": "stable-baselines3-dqn",
            "actions": {"pilot": PILOT_ACTIONS, "enemies": ENEMY_ACTIONS},
            "features": FEATURES,
            "netArch": DQN_NET_ARCH,
            "roundNumber": round_number,
            "phaseNumber": phase_number,
            "checkpointCounts": {
                "pilot": len(retained_checkpoints.get("pilot", [])),
                "enemies": len(retained_checkpoints.get("enemies", [])),
            },
            "totalGenerationCounts": generation_counts(history),
            "checkpointRetention": self.retention.to_json(),
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
    required_new_balanced_rounds: int = 0,
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
    curriculum_waves: int = 3,
    candidate_spawns: int = 1,
    checkpoint_retention: CheckpointRetention | None = None,
) -> tuple[DQN, DQN, dict[str, object]]:
    if timesteps_per_round is not None:
        phase_timesteps = timesteps_per_round
    if balanced_rounds is not None and balanced_rounds < 2:
        raise ValueError("balanced_rounds must be at least 2 so both roles can get a checkpoint.")
    required_new_balanced_rounds = max(0, int(required_new_balanced_rounds))
    train_workers = max(1, int(train_workers))
    eval_workers = max(1, int(eval_workers))
    curriculum_waves = max(1, int(curriculum_waves))
    candidate_spawns = max(1, int(candidate_spawns))
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
    checkpoint_retention = checkpoint_retention or CheckpointRetention()
    checkpoint_store = TrainingCheckpointStore(checkpoint_dir, checkpoint_retention) if checkpoint_dir is not None else None
    if resume:
        if checkpoint_store is None:
            raise RuntimeError("--resume requires checkpoint_dir.")
        loaded = checkpoint_store.load()
        pilot_model = loaded.pilot_model
        enemy_model = loaded.enemy_model
        history = loaded.history
        checkpoints = retain_checkpoint_sets(loaded.checkpoints, checkpoint_retention)
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
        "requiredNewBalancedRounds": required_new_balanced_rounds,
        "balanceTolerance": balance_tolerance,
        "balancePatience": balance_patience,
        "balanceMinWinRate": balance_min_win_rate,
        "rounds": rounds,
        "trainWorkers": train_workers,
        "evalWorkers": eval_workers,
        "curriculumWaves": curriculum_waves,
        "candidateSpawns": candidate_spawns,
        "checkpointRetention": checkpoint_retention.to_json(),
    }

    current_phase_number = phase_number_offset

    def run_generation(role: str, phase_number: int, phase_iteration: int) -> bool:
        nonlocal pilot_model, enemy_model, round_number, checkpoints, current_phase_number

        round_number += 1
        current_phase_number = phase_number
        phase_seed = seed + phase_number * 1000 + phase_iteration * 97
        candidate_results: list[CandidateResult] = []
        selected_candidate = 1
        if role == "pilot":
            opponent_spec = enemy_policy_spec(enemy_model)
            if candidate_spawns > 1:
                with tempfile.TemporaryDirectory(prefix="galagai-pilot-candidates.") as tmpdir:
                    candidate_dir = Path(tmpdir)
                    base_model_path, base_replay_path = save_candidate_base_model(pilot_model, candidate_dir, "pilot")
                    tasks = [
                        (
                            "pilot",
                            spawn_index,
                            base_model_path,
                            base_replay_path,
                            opponent_spec,
                            phase_seed + spawn_index * 10_003,
                            seed + round_number * 3000,
                            phase_timesteps,
                            max_steps,
                            train_workers,
                            eval_episodes,
                            eval_workers,
                            curriculum_waves,
                            str(candidate_dir),
                        )
                        for spawn_index in range(1, candidate_spawns + 1)
                    ]
                    with ProcessPoolExecutor(max_workers=candidate_spawns) as executor:
                        candidate_results = list(executor.map(train_candidate_from_files, tasks))
                    selected = best_candidate_result(candidate_results, "pilot")
                    selected_candidate = selected.spawn_index
                    pilot_model = load_candidate_model(selected)
                    metrics = selected.metrics
            else:
                pilot_model = train_role_model(
                    role="pilot",
                    model=pilot_model,
                    opponent_spec=opponent_spec,
                    seed=phase_seed,
                    phase_timesteps=phase_timesteps,
                    max_steps=max_steps,
                    train_workers=train_workers,
                    curriculum_waves=curriculum_waves,
                )
                metrics = evaluate_current_matchup(
                    seed=seed + round_number * 3000,
                    pilot_model=pilot_model,
                    enemy_model=enemy_model,
                    eval_episodes=eval_episodes,
                    max_steps=max_steps,
                    eval_workers=eval_workers,
                    curriculum_waves=curriculum_waves,
                )
            trained = "pilot"
        else:
            opponent_spec = pilot_policy_spec(pilot_model)
            if candidate_spawns > 1:
                with tempfile.TemporaryDirectory(prefix="galagai-enemy-candidates.") as tmpdir:
                    candidate_dir = Path(tmpdir)
                    base_model_path, base_replay_path = save_candidate_base_model(enemy_model, candidate_dir, "enemies")
                    tasks = [
                        (
                            "enemies",
                            spawn_index,
                            base_model_path,
                            base_replay_path,
                            opponent_spec,
                            phase_seed + spawn_index * 10_003,
                            seed + round_number * 3000,
                            phase_timesteps,
                            max_steps,
                            train_workers,
                            eval_episodes,
                            eval_workers,
                            curriculum_waves,
                            str(candidate_dir),
                        )
                        for spawn_index in range(1, candidate_spawns + 1)
                    ]
                    with ProcessPoolExecutor(max_workers=candidate_spawns) as executor:
                        candidate_results = list(executor.map(train_candidate_from_files, tasks))
                    selected = best_candidate_result(candidate_results, "enemies")
                    selected_candidate = selected.spawn_index
                    enemy_model = load_candidate_model(selected)
                    metrics = selected.metrics
            else:
                enemy_model = train_role_model(
                    role="enemies",
                    model=enemy_model,
                    opponent_spec=opponent_spec,
                    seed=phase_seed,
                    phase_timesteps=phase_timesteps,
                    max_steps=max_steps,
                    train_workers=train_workers,
                    curriculum_waves=curriculum_waves,
                )
                metrics = evaluate_current_matchup(
                    seed=seed + round_number * 3000,
                    pilot_model=pilot_model,
                    enemy_model=enemy_model,
                    eval_episodes=eval_episodes,
                    max_steps=max_steps,
                    eval_workers=eval_workers,
                    curriculum_waves=curriculum_waves,
                )
            trained = "enemies"

        dominance_metric = "pilotWinRate" if trained == "pilot" else "enemyWinRate"
        dominance_reached = float(metrics[dominance_metric]) >= dominance_threshold
        generation = role_generation_count(history, trained) + 1
        round_metrics = {
            "round": round_number,
            "phase": phase_number,
            "phaseIteration": phase_iteration,
            "trained": trained,
            "generation": generation,
            "dominanceMetric": dominance_metric,
            "dominanceThreshold": dominance_threshold,
            "dominanceReached": dominance_reached,
            "candidateSpawns": candidate_spawns,
            "selectedCandidate": selected_candidate,
            **metrics,
        }
        if candidate_results:
            round_metrics["candidateMetrics"] = candidate_metric_summary(candidate_results)
        history.append(round_metrics)

        if trained == "pilot" and pilot_model is not None:
            checkpoints["pilot"].append(
                checkpoint_entry(
                    role="pilot",
                    model=pilot_model,
                    version_id=generation,
                    metrics=round_metrics,
                )
            )
        elif trained == "enemies" and enemy_model is not None:
            checkpoints["enemies"].append(
                checkpoint_entry(
                    role="enemies",
                    model=enemy_model,
                    version_id=generation,
                    metrics=round_metrics,
                )
            )
        checkpoints = retain_checkpoint_sets(checkpoints, checkpoint_retention)
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
        totals = generation_counts(history)
        progress_tracker.update(round_metrics, totals["pilot"], totals["enemies"])
        return dominance_reached

    try:
        if balanced_rounds is not None:
            phase_number = phase_number_offset
            while len(history) < balanced_rounds:
                if balanced_stop_reached_after_required_rounds(
                    history,
                    completed_generations=completed_generations,
                    required_new_balanced_rounds=required_new_balanced_rounds,
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
                    if dominance_reached or balanced_stop_reached_after_required_rounds(
                        history,
                        completed_generations=completed_generations,
                        required_new_balanced_rounds=required_new_balanced_rounds,
                        min_balanced_rounds=min_balanced_rounds,
                        balance_patience=balance_patience,
                        dominance_threshold=dominance_threshold,
                        balance_tolerance=balance_tolerance,
                        balance_min_win_rate=balance_min_win_rate,
                    ):
                        break
        elif generations_per_side is not None:
            phase_number = phase_number_offset
            totals = generation_counts(history)
            role = "pilot" if totals["pilot"] <= totals["enemies"] else "enemies"
            while totals["pilot"] < generations_per_side or totals["enemies"] < generations_per_side:
                totals = generation_counts(history)
                if totals[role] >= generations_per_side:
                    role = "enemies" if role == "pilot" else "pilot"
                    continue
                phase_number += 1
                for phase_iteration in range(1, max_phase_iterations + 1):
                    totals = generation_counts(history)
                    if totals[role] >= generations_per_side:
                        break
                    dominance_reached = run_generation(role, phase_number, phase_iteration)
                    if dominance_reached:
                        break
                totals = generation_counts(history)
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

    checkpoints = retain_checkpoint_sets(checkpoints, checkpoint_retention)
    if checkpoint_store is not None and checkpoint_retention.mode != "all":
        checkpoint_store.save(
            pilot_model=pilot_model,
            enemy_model=enemy_model,
            history=history,
            checkpoints=checkpoints,
            round_number=round_number,
            phase_number=current_phase_number,
            config=checkpoint_config,
        )

    return pilot_model, enemy_model, {
        "type": "stable-baselines3-dqn-galagAI-dominance-self-play",
        "rounds": history,
        "latest": history[-1],
        "checkpoints": checkpoints,
        "retainedCheckpointCounts": {
            "pilot": len(checkpoints["pilot"]),
            "enemies": len(checkpoints["enemies"]),
        },
        "totalGenerationCounts": generation_counts(history),
        "checkpointRetention": checkpoint_retention.to_json(),
        "cycles": cycles,
        "generationsPerSide": generations_per_side,
        "balancedRounds": balanced_rounds,
        "minBalancedRounds": min_balanced_rounds,
        "requiredNewBalancedRounds": required_new_balanced_rounds,
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
        "curriculumWaves": curriculum_waves,
        "candidateSpawns": candidate_spawns,
        "environment": {
            "name": "HeadlessGalagai",
            "openingEnemyPolicy": "role-gated bootstrap: bees drift/drop, butterflies and bosses can shoot after wave one",
            "curriculumWaves": curriculum_waves,
            "dropCooldownSeconds": DROP_COOLDOWN_SECONDS,
            "enemyShotCooldownSeconds": ENEMY_SHOT_COOLDOWN_SECONDS,
            "actionDtSeconds": ACTION_DT,
            "antiDropSpam": "invalid drops are ignored and penalized",
            "npcAccessibility": "pilot observes dangerous enemy shots; enemies observe dangerous pilot bullets and role counts",
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
    curriculum_waves: int = 1,
) -> dict[str, object]:
    return evaluate_policy_specs(
        seed=seed,
        pilot_spec=pilot_policy_spec(pilot_model),
        enemy_spec=enemy_policy_spec(enemy_model),
        episodes=eval_episodes,
        max_steps=max_steps,
        workers=eval_workers,
        curriculum_waves=curriculum_waves,
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


def run_episode(
    seed: int,
    pilot_policy: Policy,
    enemy_policy: Policy,
    max_steps: int,
    curriculum_waves: int = 1,
) -> EpisodeResult:
    game = HeadlessGalagai(seed=seed, max_steps=max_steps, max_start_wave=curriculum_waves)
    observation = game.reset(seed=seed)
    done = False
    info: dict[str, object] = {}
    while not done:
        pilot_action = pilot_policy.act(observation)
        enemy_actions = game.enemy_policy_actions(enemy_policy)
        observation, _, _, done, info = game.step(pilot_action, enemy_actions)
    drop_attempts = max(1, int(info.get("dropAttempts", 0)))
    pilot_fires = int(info.get("pilotFires", 0))
    pilot_hits = int(info.get("pilotHits", 0))
    wave_cleared = int(info.get("wave", 1)) > int(info.get("startWave", 1))
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


def run_episode_from_specs(args: tuple[int, PolicySpec, PolicySpec, int, int]) -> EpisodeResult:
    seed, pilot_spec, enemy_spec, max_steps, curriculum_waves = args
    return run_episode(seed, policy_from_spec(pilot_spec), policy_from_spec(enemy_spec), max_steps, curriculum_waves)


def evaluate_policy_specs(
    *,
    seed: int,
    pilot_spec: PolicySpec,
    enemy_spec: PolicySpec,
    episodes: int,
    max_steps: int,
    workers: int,
    curriculum_waves: int = 1,
) -> dict[str, object]:
    curriculum_waves = max(1, int(curriculum_waves))
    if workers > 1 and episodes > 1:
        tasks = [(seed + episode, pilot_spec, enemy_spec, max_steps, curriculum_waves) for episode in range(episodes)]
        with ProcessPoolExecutor(max_workers=min(workers, episodes)) as executor:
            results = list(executor.map(run_episode_from_specs, tasks))
    else:
        pilot_policy = policy_from_spec(pilot_spec)
        enemy_policy = policy_from_spec(enemy_spec)
        results = [
            run_episode(seed + episode, pilot_policy, enemy_policy, max_steps, curriculum_waves)
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
    parser.add_argument(
        "--required-new-balanced-rounds",
        type=int,
        default=0,
        help="Require this many new balanced generations after resume before the balance stop gate can end training.",
    )
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
    parser.add_argument(
        "--curriculum-waves",
        type=int,
        default=3,
        help="Randomize episode starts across waves 1..N so pilot/enemy policies train against later enemy roles.",
    )
    parser.add_argument(
        "--candidate-spawns",
        type=int,
        default=1,
        help="Train this many independent candidates per generation in parallel and keep the best evaluated candidate.",
    )
    parser.add_argument(
        "--checkpoint-retention",
        choices=("all", "tiered"),
        default="all",
        help="Which exported checkpoint JSON files to keep. 'tiered' keeps latest N, every 2 through 100, every 10 through 1000, and every 100 after.",
    )
    parser.add_argument("--keep-latest-versions", type=int, default=RETENTION_LATEST_DEFAULT)
    args = parser.parse_args()
    retention = CheckpointRetention(mode=args.checkpoint_retention, keep_latest=args.keep_latest_versions)

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
        required_new_balanced_rounds=args.required_new_balanced_rounds,
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
        curriculum_waves=args.curriculum_waves,
        candidate_spawns=args.candidate_spawns,
        checkpoint_retention=retention,
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
                "requiredNewBalancedRounds": self_play["requiredNewBalancedRounds"],
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
                "curriculumWaves": self_play["curriculumWaves"],
                "candidateSpawns": self_play["candidateSpawns"],
                "checkpointRetention": self_play["checkpointRetention"],
                "totalGenerationCounts": self_play["totalGenerationCounts"],
                "retainedCheckpointCounts": self_play["retainedCheckpointCounts"],
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
