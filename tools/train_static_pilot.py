"""Train RL pilot and enemy policies for the static GalagAI demo.

The trainer uses a headless Gymnasium version of the browser game loop and
alternates Stable-Baselines3 DQN training between the pilot and enemy roles. The
exported JSON contains both learned Q networks so GitHub Pages can run the
latest pilot and enemy policies without loading Python dependencies.
"""

from __future__ import annotations

import argparse
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
        threat_dx = float(observation[2])
        drop_ready = observation[7] > 0.5
        shot_ready = observation[8] > 0.5
        fleet_y = float(observation[9])
        if shot_ready and abs(threat_dx) < 0.18:
            return 3
        if drop_ready and fleet_y < 0.42 and abs(threat_dx) > 0.28:
            return 1
        return 2 if threat_dx > 0 else 0


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
                self.enemy_drop_cooldown = 1.08
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
                    self.enemy_shot_cooldown = 0.42
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
        reward = events.score_gain / 50.0
        reward += 4.0 if events.wave_cleared else 0.0
        reward -= 5.0 * max(0, events.life_loss)
        reward -= 0.10 if events.pilot_missed else 0.0
        reward -= 0.01
        if done and self.lives <= 0:
            reward -= 4.0
        return float(reward)

    def enemy_reward(self, events: StepEvents, done: bool) -> float:
        reward = 5.0 * max(0, events.life_loss)
        reward -= 1.2 * max(0, events.aliens_destroyed)
        reward -= events.score_gain / 70.0
        reward += 0.015 * self.live_alien_count / max(1, len(self.aliens))
        reward -= 0.08 if events.valid_drop else 0.0
        reward -= 0.90 if events.invalid_drop else 0.0
        reward -= 0.25 if events.invalid_fire else 0.0
        reward += 0.04 if events.enemy_fired else 0.0
        if done and self.lives <= 0:
            reward += 4.0
        elif done and self.lives > 0:
            reward -= 1.5
        return float(reward)

    def winner(self, done: bool) -> str:
        if not done:
            return "none"
        if self.lives <= 0:
            return "enemies"
        if self.score >= 300 or self.wave > 1:
            return "pilot"
        return "timeout"

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
        return next_observation, pilot_reward, done, False, info


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
        return next_observation, enemy_reward, done, False, info


def make_dqn(env: gym.Env, seed: int, learning_rate: float) -> DQN:
    return DQN(
        "MlpPolicy",
        Monitor(env),
        learning_rate=learning_rate,
        buffer_size=20_000,
        learning_starts=250,
        batch_size=64,
        gamma=0.96,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=350,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [64, 64]},
        seed=seed,
        verbose=0,
    )


def train_self_play(
    *,
    seed: int,
    rounds: int,
    timesteps_per_round: int,
    eval_episodes: int,
    max_steps: int,
) -> tuple[DQN, DQN, dict[str, object]]:
    pilot_model: DQN | None = None
    enemy_model: DQN | None = None
    history: list[dict[str, object]] = []

    for round_number in range(1, rounds + 1):
        if round_number % 2 == 1:
            enemy_policy: Policy = SB3Policy(enemy_model) if enemy_model is not None else HeuristicEnemyPolicy()
            env = PilotTrainingEnv(enemy_policy, seed=seed + round_number * 1000, max_steps=max_steps)
            if pilot_model is None:
                pilot_model = make_dqn(env, seed=seed, learning_rate=8e-4)
            else:
                pilot_model.set_env(Monitor(env))
            pilot_model.learn(total_timesteps=timesteps_per_round, reset_num_timesteps=False, progress_bar=False)
            trained = "pilot"
        else:
            pilot_policy: Policy = SB3Policy(pilot_model) if pilot_model is not None else HeuristicPilotPolicy()
            env = EnemyTrainingEnv(pilot_policy, seed=seed + round_number * 1000, max_steps=max_steps)
            if enemy_model is None:
                enemy_model = make_dqn(env, seed=seed + 1, learning_rate=8e-4)
            else:
                enemy_model.set_env(Monitor(env))
            enemy_model.learn(total_timesteps=timesteps_per_round, reset_num_timesteps=False, progress_bar=False)
            trained = "enemies"

        if pilot_model is not None and enemy_model is not None:
            metrics = evaluate(seed + round_number * 3000, SB3Policy(pilot_model), SB3Policy(enemy_model), eval_episodes, max_steps)
        elif pilot_model is not None:
            metrics = evaluate(seed + round_number * 3000, SB3Policy(pilot_model), HeuristicEnemyPolicy(), eval_episodes, max_steps)
        else:
            metrics = evaluate(seed + round_number * 3000, HeuristicPilotPolicy(), SB3Policy(enemy_model), eval_episodes, max_steps)
        history.append({"round": round_number, "trained": trained, **metrics})

    if pilot_model is None or enemy_model is None:
        raise RuntimeError("Training requires at least one pilot round and one enemy round.")

    return pilot_model, enemy_model, {
        "type": "stable-baselines3-dqn-galagAI-self-play",
        "rounds": history,
        "latest": history[-1],
        "environment": {
            "name": "HeadlessGalagai",
            "dropCooldownSeconds": 1.08,
            "enemyShotCooldownSeconds": 0.42,
            "actionDtSeconds": ACTION_DT,
            "antiDropSpam": "invalid drops are ignored and penalized",
        },
    }


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
    payload = {
        "model": "sb3-dqn-pilot",
        "version": 4,
        "algorithm": "stable-baselines3-dqn",
        "actions": PILOT_ACTIONS,
        "features": FEATURES,
        "network": export_network(pilot_model),
        "enemies": {
            "model": "sb3-dqn-enemies",
            "actions": ENEMY_ACTIONS,
            "features": FEATURES,
            "network": export_network(enemy_model),
            "constraints": {
                "dropCooldownSeconds": 1.08,
                "shotCooldownSeconds": 0.42,
                "invalidDropPenalty": 0.9,
            },
        },
        "metrics": {
            "rlAlgorithm": "stable-baselines3-dqn",
            "evalAccuracy": float(latest["pilotWinRate"]),
            "enemyWinRate": float(latest["enemyWinRate"]),
            "enemyDropRate": float(latest["enemyDropRate"]),
            "invalidDropRate": float(latest["invalidDropRate"]),
            "enemyFireRate": float(latest["enemyFireRate"]),
            "selfPlay": self_play,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train static GalagAI pilot/enemy SB3 DQN policies.")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--timesteps-per-round", type=int, default=1600)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--out", type=Path, default=Path("js/galagai-model.json"))
    args = parser.parse_args()

    pilot_model, enemy_model, self_play = train_self_play(
        seed=args.seed,
        rounds=args.rounds,
        timesteps_per_round=args.timesteps_per_round,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
    )
    write_model(args.out, pilot_model, enemy_model, self_play)
    print(
        json.dumps(
            {
                "model": str(args.out),
                "algorithm": "stable-baselines3-dqn",
                "rounds": args.rounds,
                "timestepsPerRound": args.timesteps_per_round,
                "evalEpisodes": args.eval_episodes,
                "selfPlayLatest": self_play["latest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
