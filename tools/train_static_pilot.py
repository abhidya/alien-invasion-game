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
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from tqdm.auto import tqdm


PILOT_ACTIONS = ["left", "right", "fire", "stay"]
ENEMY_ACTIONS = ["drift_left", "drop", "drift_right", "fire"]
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
ENEMY_SHOT_COOLDOWN_SECONDS = 0.42
INVALID_DROP_PENALTY = 0.90
MODEL_SCHEMA_VERSION = 6
DQN_NET_ARCH = [64, 64]
MODEL_FILE_DIR = "galagai-models"


@dataclass
class Actor:
    x: float
    y: float
    width: float
    height: float
    alive: bool = True

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
    lives_left: int


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
        shot_ready = observation[8] > 0.5
        fleet_y = float(observation[9])
        if shot_ready and abs(target_dx) < 0.18:
            return 3
        if drop_ready and fleet_y < 0.42 and abs(target_dx) > 0.34:
            return 1
        return 0 if target_dx > 0 else 2


class OpeningEnemyPolicy:
    def act(self, observation: np.ndarray) -> int:
        target_dx = float(observation[0])
        return 0 if target_dx > 0 else 2


class SB3Policy:
    def __init__(self, model: DQN):
        self.model = model

    def act(self, observation: np.ndarray) -> int:
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)


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
                aliens.append(Actor(start_x + col * gap_x, 74.0 + row * gap_y, ALIEN_WIDTH, ALIEN_HEIGHT))
        return aliens

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
                1.0 if self.enemy_shot_cooldown <= 0 else 0.0,
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
            self.drop_attempts += 1
            if self.enemy_drop_cooldown <= 0:
                self.drop_fleet()
                self.enemy_drop_cooldown = DROP_COOLDOWN_SECONDS
                valid_drop = True
            else:
                self.invalid_drops += 1
                invalid_drop = True
        elif enemy_action == 2:
            self.fleet_direction = 1
        elif enemy_action == 3:
            if self.enemy_shot_cooldown <= 0:
                enemy_fired = self.fire_enemy_shot()
                if enemy_fired:
                    self.enemy_shot_cooldown = ENEMY_SHOT_COOLDOWN_SECONDS
                    self.enemy_fires += 1
                else:
                    invalid_fire = True
            else:
                invalid_fire = True
        else:
            raise ValueError(f"Unknown enemy action {enemy_action}.")

        self.update_projectiles()
        self.update_aliens()
        hit_events = self.resolve_collisions()

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

    def fire_enemy_shot(self) -> bool:
        live_aliens = [alien for alien in self.aliens if alien.alive]
        if not live_aliens:
            return False
        alien = min(live_aliens, key=lambda item: abs(item.center_x - self.ship.center_x))
        self.enemy_shots.append(Shot(alien.center_x - 3.0, alien.bottom, 6.0, 16.0, ENEMY_SHOT_SPEED + self.wave * 20.0))
        return True

    def drop_fleet(self) -> None:
        for alien in self.aliens:
            if alien.alive:
                alien.y += FLEET_DROP * 0.65

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
        reward += 7.0 if events.wave_cleared else 0.0
        reward -= 7.0 * max(0, events.life_loss)
        reward += 0.08 if events.pilot_aligned_action else 0.0
        reward += 0.24 if events.pilot_aligned_fire else 0.0
        reward -= 0.18 if events.pilot_bad_fire else 0.0
        reward -= 0.20 if events.pilot_missed else 0.0
        reward -= 0.002
        if done:
            winner = self.winner(done)
            if winner == "pilot":
                reward += 6.0
            elif winner == "enemies":
                reward -= 6.0
            else:
                reward -= 1.0
        return float(reward)

    def enemy_reward(self, events: StepEvents, done: bool) -> float:
        reward = 7.0 * max(0, events.life_loss)
        reward -= 1.45 * max(0, events.aliens_destroyed)
        reward -= events.score_gain / 55.0
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
        if self.live_alien_count == 0 or self.score >= 300 or self.wave > 1:
            return "pilot"
        if self.steps >= self.max_steps and self.lives >= 2 and self.score >= 100:
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
        if action == 2:
            return target_dx < -0.06
        if action == 3:
            return abs(target_dx) < 0.18
        if action == 1:
            return fleet_y < 0.42 and abs(target_dx) > 0.34
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
        return next_observation, pilot_reward, terminated, truncated, info


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
        return next_observation, enemy_reward, terminated, truncated, info


def make_dqn(env: gym.Env, seed: int, learning_rate: float) -> DQN:
    return DQN(
        "MlpPolicy",
        Monitor(env),
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
    def __init__(self, enabled: bool, total: int):
        self.bar = tqdm(
            total=total,
            desc="self-play generations",
            unit="gen",
            dynamic_ncols=True,
            disable=not enabled,
        )

    def update(self, metrics: dict[str, object], pilot_count: int, enemy_count: int) -> None:
        if self.bar.disable:
            return
        self.bar.set_postfix(
            side=metrics["trained"],
            pilot=pilot_count,
            enemies=enemy_count,
            p_win=f"{float(metrics['pilotWinRate']):.2f}",
            e_win=f"{float(metrics['enemyWinRate']):.2f}",
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


def progress_total(cycles: int, max_phase_iterations: int, generations_per_side: int | None, rounds: int | None) -> int:
    if generations_per_side is not None:
        return generations_per_side * 2
    if rounds is not None:
        return rounds
    return cycles * 2 * max_phase_iterations


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
    rounds: int | None = None,
    timesteps_per_round: int | None = None,
    progress: bool = False,
) -> tuple[DQN, DQN, dict[str, object]]:
    if timesteps_per_round is not None:
        phase_timesteps = timesteps_per_round
    if generations_per_side is not None:
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
    progress_tracker = TrainingProgress(
        progress,
        progress_total(cycles, max_phase_iterations, generations_per_side, rounds),
    )

    def run_generation(role: str, phase_number: int, phase_iteration: int) -> bool:
        nonlocal pilot_model, enemy_model, round_number

        round_number += 1
        phase_seed = seed + phase_number * 1000 + phase_iteration * 97
        if role == "pilot":
            enemy_policy: Policy = SB3Policy(enemy_model) if enemy_model is not None else OpeningEnemyPolicy()
            env = PilotTrainingEnv(enemy_policy, seed=phase_seed, max_steps=max_steps)
            if pilot_model is None:
                pilot_model = make_dqn(env, seed=seed, learning_rate=8e-4)
            else:
                pilot_model.set_env(Monitor(env))
            pilot_model.learn(total_timesteps=phase_timesteps, reset_num_timesteps=False, progress_bar=False)
            trained = "pilot"
        else:
            pilot_policy: Policy = SB3Policy(pilot_model) if pilot_model is not None else HeuristicPilotPolicy()
            env = EnemyTrainingEnv(pilot_policy, seed=phase_seed, max_steps=max_steps)
            if enemy_model is None:
                enemy_model = make_dqn(env, seed=seed + 1, learning_rate=8e-4)
            else:
                enemy_model.set_env(Monitor(env))
            enemy_model.learn(total_timesteps=phase_timesteps, reset_num_timesteps=False, progress_bar=False)
            trained = "enemies"

        metrics = evaluate_current_matchup(
            seed=seed + round_number * 3000,
            pilot_model=pilot_model,
            enemy_model=enemy_model,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
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
        progress_tracker.update(round_metrics, len(checkpoints["pilot"]), len(checkpoints["enemies"]))
        return dominance_reached

    try:
        if generations_per_side is not None:
            phase_number = 0
            role = "pilot"
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
            for phase_number, role in enumerate(phase_roles, start=1):
                for phase_iteration in range(1, max_phase_iterations + 1):
                    dominance_reached = run_generation(role, phase_number, phase_iteration)
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
        "phaseTimesteps": phase_timesteps,
        "maxPhaseIterations": max_phase_iterations,
        "dominanceThreshold": dominance_threshold,
        "netArch": DQN_NET_ARCH,
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
) -> dict[str, object]:
    pilot_policy: Policy = SB3Policy(pilot_model) if pilot_model is not None else HeuristicPilotPolicy()
    enemy_policy: Policy = SB3Policy(enemy_model) if enemy_model is not None else OpeningEnemyPolicy()
    return evaluate(seed, pilot_policy, enemy_policy, eval_episodes, max_steps)


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


def evaluate(seed: int, pilot_policy: Policy, enemy_policy: Policy, episodes: int, max_steps: int) -> dict[str, object]:
    results: list[EpisodeResult] = []
    for episode in range(episodes):
        game = HeadlessGalagai(seed=seed + episode, max_steps=max_steps)
        observation = game.reset(seed=seed + episode)
        done = False
        info: dict[str, object] = {}
        while not done:
            pilot_action = pilot_policy.act(observation)
            enemy_action = enemy_policy.act(observation)
            observation, _, _, done, info = game.step(pilot_action, enemy_action)
        drop_attempts = max(1, int(info.get("dropAttempts", 0)))
        results.append(
            EpisodeResult(
                winner=str(info.get("winner", "none")),
                score=int(info.get("score", 0)),
                wave=int(info.get("wave", 1)),
                steps=game.steps,
                enemy_drop_rate=float(info.get("dropAttempts", 0)) / max(1, game.steps),
                invalid_drop_rate=float(info.get("invalidDrops", 0)) / drop_attempts,
                enemy_fire_rate=float(info.get("enemyFires", 0)) / max(1, game.steps),
                pilot_fire_rate=float(info.get("pilotFires", 0)) / max(1, game.steps),
                lives_left=int(info.get("lives", 0)),
            )
        )

    pilot_wins = sum(1 for result in results if result.winner == "pilot")
    enemy_wins = sum(1 for result in results if result.winner == "enemies")
    return {
        "pilotWinRate": round(pilot_wins / max(1, episodes), 4),
        "enemyWinRate": round(enemy_wins / max(1, episodes), 4),
        "averageScore": round(float(np.mean([result.score for result in results])), 2),
        "averageWave": round(float(np.mean([result.wave for result in results])), 2),
        "averageSteps": round(float(np.mean([result.steps for result in results])), 2),
        "enemyDropRate": round(float(np.mean([result.enemy_drop_rate for result in results])), 4),
        "invalidDropRate": round(float(np.mean([result.invalid_drop_rate for result in results])), 4),
        "enemyFireRate": round(float(np.mean([result.enemy_fire_rate for result in results])), 4),
        "pilotFireRate": round(float(np.mean([result.pilot_fire_rate for result in results])), 4),
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
    parser.add_argument("--rounds", type=int, default=None, help="Deprecated fixed alternation phase count.")
    parser.add_argument("--timesteps-per-round", type=int, default=None, help="Deprecated alias for --phase-timesteps.")
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--out", type=Path, default=Path("js/galagai-model.json"))
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output.")
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
        rounds=args.rounds,
        timesteps_per_round=args.timesteps_per_round,
        progress=not args.no_progress,
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
                "phaseTimesteps": self_play["phaseTimesteps"],
                "dominanceThreshold": self_play["dominanceThreshold"],
                "maxPhaseIterations": self_play["maxPhaseIterations"],
                "netArch": self_play["netArch"],
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
