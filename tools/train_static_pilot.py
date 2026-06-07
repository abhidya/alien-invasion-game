"""Train NumPy RL policies for the static GalagAI demo.

This script uses a tiny feature-based self-play environment so it runs on the
repo's current Python 3.14 workspace without TensorFlow or PyTorch. The exported
JSON drives both the browser pilot and the browser enemies.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alien_invasion.DQN import DQNAgent, DQNConfig


ACTIONS = ["left", "right", "fire", "stay"]
ENEMY_ACTIONS = ["drift_left", "drop", "drift_right", "fire"]
FEATURES = [
    "bias",
    "target_dx",
    "abs_target_dx",
    "threat_dx",
    "threat_y",
    "bullet_ready",
    "alien_count",
    "wave",
]


@dataclass
class StaticEpisodeResult:
    winner: str
    score: int
    wave: int
    steps: int
    enemy_pressure: float


class StaticSelfPlayEnv:
    """Small deterministic-enough arcade model for fast local RL training."""

    def __init__(self, seed: int, max_steps: int = 80):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> np.ndarray:
        self.target_dx = float(self.rng.uniform(-0.9, 0.9))
        self.threat_dx = float(self.rng.uniform(-0.85, 0.85))
        self.threat_y = float(self.rng.uniform(0.0, 0.35))
        self.bullet_ready = 1.0
        self.bullet_cooldown = 0
        self.alien_count = 1.0
        self.wave = 1
        self.lives = 3
        self.score = 0
        self.steps = 0
        self.enemy_pressure = 0.35
        return self.features()

    def features(self) -> np.ndarray:
        return np.array(
            [
                1.0,
                self.target_dx,
                abs(self.target_dx),
                self.threat_dx,
                self.threat_y,
                self.bullet_ready,
                self.alien_count,
                min(self.wave / 10.0, 1.0),
            ],
            dtype=np.float32,
        )

    def step(self, pilot_action: int, enemy_action: int) -> tuple[np.ndarray, float, float, bool, dict[str, object]]:
        self.steps += 1
        pilot_reward_value = -0.01
        enemy_reward_value = 0.005

        if pilot_action == 0:
            self.target_dx = self._clamp(self.target_dx + 0.18)
            self.threat_dx = self._clamp(self.threat_dx + 0.20)
        elif pilot_action == 1:
            self.target_dx = self._clamp(self.target_dx - 0.18)
            self.threat_dx = self._clamp(self.threat_dx - 0.20)

        if enemy_action == 0:
            self.target_dx = self._clamp(self.target_dx - 0.05)
            self.threat_dx = self._clamp(self.threat_dx - 0.16)
            self.enemy_pressure = self._clamp(self.enemy_pressure + 0.004, 0.0, 1.0)
        elif enemy_action == 1:
            self.threat_y = self._clamp(self.threat_y + 0.11, 0.0, 1.2)
            self.enemy_pressure = self._clamp(self.enemy_pressure + 0.008, 0.0, 1.0)
        elif enemy_action == 2:
            self.target_dx = self._clamp(self.target_dx + 0.05)
            self.threat_dx = self._clamp(self.threat_dx + 0.16)
            self.enemy_pressure = self._clamp(self.enemy_pressure + 0.004, 0.0, 1.0)
        elif enemy_action == 3:
            self.threat_y = max(self.threat_y, 0.58)
            self.enemy_pressure = self._clamp(self.enemy_pressure + 0.012, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown enemy action {enemy_action}.")

        if pilot_action == 2:
            if self.bullet_ready and abs(self.target_dx) < 0.15:
                self.score += 50 * self.wave
                self.alien_count = max(0.0, self.alien_count - 0.16)
                pilot_reward_value += 2.0
                enemy_reward_value -= 1.2
                self.target_dx = float(self.rng.uniform(-0.9, 0.9))
            elif self.bullet_ready:
                pilot_reward_value -= 0.18
                enemy_reward_value += 0.08
            self.bullet_ready = 0.0
            self.bullet_cooldown = 3

        if self.bullet_cooldown > 0:
            self.bullet_cooldown -= 1
        else:
            self.bullet_ready = 1.0

        self.threat_y += 0.065 + self.enemy_pressure * 0.035
        if self.threat_y >= 1.0:
            if abs(self.threat_dx) < 0.17:
                self.lives -= 1
                pilot_reward_value -= 3.0
                enemy_reward_value += 3.0
            self.threat_dx = float(self.rng.uniform(-0.85, 0.85))
            self.threat_y = float(self.rng.uniform(0.0, 0.22))

        self.target_dx = self._clamp(self.target_dx + float(self.rng.normal(0.0, 0.025)))
        self.threat_dx = self._clamp(self.threat_dx + float(self.rng.normal(0.0, 0.03)))

        done = False
        winner = "timeout"
        if self.alien_count <= 0.0:
            done = True
            winner = "pilot"
            pilot_reward_value += 4.0
            enemy_reward_value -= 2.5
        elif self.lives <= 0:
            done = True
            winner = "enemies"
            pilot_reward_value -= 4.0
            enemy_reward_value += 4.0
        elif self.steps >= self.max_steps:
            done = True
            winner = "pilot" if self.score >= 150 else "enemies"

        info = {
            "winner": winner,
            "score": self.score,
            "wave": self.wave,
            "enemyPressure": self.enemy_pressure,
            "steps": self.steps,
        }
        return self.features(), float(pilot_reward_value), float(enemy_reward_value), done, info

    @staticmethod
    def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
        return max(low, min(high, value))


def build_agent(action_count: int, seed: int) -> DQNAgent:
    config = DQNConfig(
        state_shape=(len(FEATURES),),
        action_count=action_count,
        include_bias=False,
        gamma=0.91,
        learning_rate=0.045,
        l2=0.0003,
        epsilon_decay_steps=12_000,
        replay_capacity=8_000,
        min_replay_size=48,
        batch_size=32,
        target_sync_interval=80,
        gradient_steps=1,
        seed_scale=0.02,
    )
    return DQNAgent(
        config=config,
        weights_path=Path(f"/tmp/galagai-static-{seed}-{action_count}.npz"),
        name="static",
        seed=seed,
        load_existing=False,
        warn_legacy=False,
    )


def run_episode(
    env: StaticSelfPlayEnv,
    pilot: DQNAgent,
    enemies: DQNAgent,
    train_role: str | None,
) -> StaticEpisodeResult:
    state = env.reset()
    done = False
    info: dict[str, object] = {}

    while not done:
        pilot_action = pilot.act(state, training=train_role == "pilot")
        enemy_action = enemies.act(state, training=train_role == "enemies")
        next_state, pilot_step_reward, enemy_step_reward, done, info = env.step(pilot_action, enemy_action)

        if train_role == "pilot":
            pilot.train_short_memory(state, pilot_action, pilot_step_reward, next_state, done)
        elif train_role == "enemies":
            enemies.train_short_memory(state, enemy_action, enemy_step_reward, next_state, done)

        state = next_state

    return StaticEpisodeResult(
        winner=str(info["winner"]),
        score=int(info["score"]),
        wave=int(info["wave"]),
        steps=int(info["steps"]),
        enemy_pressure=float(info["enemyPressure"]),
    )


def evaluate(seed: int, pilot: DQNAgent, enemies: DQNAgent, episodes: int, max_steps: int) -> dict[str, float]:
    results = [
        run_episode(StaticSelfPlayEnv(seed + index, max_steps=max_steps), pilot, enemies, train_role=None)
        for index in range(episodes)
    ]
    pilot_wins = sum(1 for result in results if result.winner == "pilot")
    return {
        "pilotWinRate": pilot_wins / max(1, episodes),
        "enemyWinRate": 1.0 - pilot_wins / max(1, episodes),
        "averageScore": float(np.mean([result.score for result in results])),
        "averageWave": float(np.mean([result.wave for result in results])),
        "enemyPressure": float(np.mean([result.enemy_pressure for result in results])),
        "averageSteps": float(np.mean([result.steps for result in results])),
    }


def train_self_play(
    *,
    seed: int,
    rounds: int,
    episodes_per_round: int,
    eval_episodes: int,
    max_steps: int,
) -> tuple[DQNAgent, DQNAgent, dict[str, object]]:
    pilot = build_agent(len(ACTIONS), seed)
    enemies = build_agent(len(ENEMY_ACTIONS), seed + 1)
    history = []

    for round_number in range(1, rounds + 1):
        trained = "pilot" if round_number % 2 == 1 else "enemies"
        for episode in range(episodes_per_round):
            env_seed = seed + round_number * 10_000 + episode
            run_episode(StaticSelfPlayEnv(env_seed, max_steps=max_steps), pilot, enemies, train_role=trained)
        pilot.sync_target_network()
        enemies.sync_target_network()
        metrics = evaluate(seed + round_number * 20_000, pilot, enemies, eval_episodes, max_steps)
        history.append(
            {
                "round": round_number,
                "trained": trained,
                "pilotWinRate": round(metrics["pilotWinRate"], 4),
                "enemyWinRate": round(metrics["enemyWinRate"], 4),
                "enemyPressure": round(metrics["enemyPressure"], 4),
                "averageScore": round(metrics["averageScore"], 2),
                "averageWave": round(metrics["averageWave"], 2),
                "averageSteps": round(metrics["averageSteps"], 2),
                "pilotEpsilon": round(pilot.epsilon, 4),
                "enemyEpsilon": round(enemies.epsilon, 4),
            }
        )

    return pilot, enemies, {
        "type": "numpy-linear-double-dqn-self-play",
        "rounds": history,
        "latest": history[-1] if history else None,
    }


def rounded_weights(agent: DQNAgent) -> list[list[float]]:
    return [[round(float(value), 6) for value in row] for row in agent.weights.tolist()]


def write_model(path: Path, pilot: DQNAgent, enemies: DQNAgent, self_play: dict[str, object]) -> None:
    latest = self_play["latest"] or {}
    payload = {
        "model": "numpy-linear-dqn-self-play",
        "version": 3,
        "actions": ACTIONS,
        "features": FEATURES,
        "weights": rounded_weights(pilot),
        "enemies": {
            "model": "numpy-linear-dqn-enemies",
            "actions": ENEMY_ACTIONS,
            "features": FEATURES,
            "weights": rounded_weights(enemies),
        },
        "metrics": {
            "rlAlgorithm": "numpy-linear-double-dqn",
            "evalAccuracy": round(float(latest.get("pilotWinRate", 0.0)), 4),
            "enemyWinRate": round(float(latest.get("enemyWinRate", 0.0)), 4),
            "selfPlay": self_play,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the static GalagAI pilot/enemy RL policies.")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--episodes-per-round", type=int, default=180)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--out", type=Path, default=Path("js/galagai-model.json"))
    args = parser.parse_args()

    pilot, enemies, self_play = train_self_play(
        seed=args.seed,
        rounds=args.rounds,
        episodes_per_round=args.episodes_per_round,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
    )
    write_model(args.out, pilot, enemies, self_play)
    print(
        json.dumps(
            {
                "model": str(args.out),
                "algorithm": "numpy-linear-double-dqn",
                "rounds": args.rounds,
                "episodesPerRound": args.episodes_per_round,
                "evalEpisodes": args.eval_episodes,
                "selfPlayLatest": self_play["latest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
