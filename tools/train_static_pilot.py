"""Train a tiny browser-pilot model for the static GalagAI demo.

The original repo's Keras DQN targets the Pygame implementation and needs a
full desktop + TensorFlow stack. This script trains a lightweight linear policy
for the dependency-free browser demo so GitHub Pages has a real model artifact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ACTIONS = ["left", "right", "fire", "stay"]
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


@dataclass(frozen=True)
class Sample:
    features: np.ndarray
    action: int


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def synthetic_state(rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Create one normalized arcade state and an expert action label."""

    target_dx = rng.uniform(-1.0, 1.0)
    threat_dx = rng.uniform(-1.0, 1.0)
    threat_y = rng.uniform(0.0, 1.0)
    bullet_ready = float(rng.random() > 0.22)
    alien_count = rng.uniform(0.05, 1.0)
    wave = rng.uniform(0.05, 1.0)

    # Expert policy: dodge imminent shots first, align to the nearest alien,
    # fire only when aligned and the cooldown is available.
    if threat_y > 0.72 and abs(threat_dx) < 0.2:
        action = 0 if threat_dx >= 0 else 1
    elif abs(target_dx) < 0.09 and bullet_ready:
        action = 2
    elif target_dx < -0.08:
        action = 0
    elif target_dx > 0.08:
        action = 1
    else:
        action = 3

    features = np.array(
        [
            1.0,
            target_dx,
            abs(target_dx),
            threat_dx,
            threat_y,
            bullet_ready,
            alien_count,
            wave,
        ],
        dtype=np.float64,
    )
    return features, action


def build_dataset(seed: int, samples: int) -> list[Sample]:
    rng = np.random.default_rng(seed)
    return [Sample(*synthetic_state(rng)) for _ in range(samples)]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train(samples: list[Sample], epochs: int, lr: float, l2: float) -> np.ndarray:
    x = np.vstack([sample.features for sample in samples])
    y = np.array([sample.action for sample in samples])
    weights = np.zeros((len(FEATURES), len(ACTIONS)), dtype=np.float64)

    for _ in range(epochs):
        probs = softmax(x @ weights)
        probs[np.arange(len(y)), y] -= 1.0
        gradient = (x.T @ probs) / len(y) + l2 * weights
        weights -= lr * gradient
    return weights


def accuracy(weights: np.ndarray, samples: list[Sample]) -> float:
    x = np.vstack([sample.features for sample in samples])
    y = np.array([sample.action for sample in samples])
    predictions = np.argmax(x @ weights, axis=1)
    return float(np.mean(predictions == y))


def confusion(weights: np.ndarray, samples: list[Sample]) -> list[list[int]]:
    matrix = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    x = np.vstack([sample.features for sample in samples])
    y = np.array([sample.action for sample in samples])
    predictions = np.argmax(x @ weights, axis=1)
    for expected, predicted in zip(y, predictions):
        matrix[int(expected), int(predicted)] += 1
    return matrix.tolist()


def write_model(path: Path, weights: np.ndarray, train_acc: float, eval_acc: float, matrix: list[list[int]]) -> None:
    payload = {
        "model": "linear-softmax-pilot",
        "version": 1,
        "actions": ACTIONS,
        "features": FEATURES,
        "weights": [[round(float(value), 6) for value in row] for row in weights.tolist()],
        "metrics": {
            "trainAccuracy": round(train_acc, 4),
            "evalAccuracy": round(eval_acc, 4),
            "confusion": matrix,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the static GalagAI browser pilot.")
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--eval-samples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=0.85)
    parser.add_argument("--l2", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--out", type=Path, default=Path("js/galagai-model.json"))
    args = parser.parse_args()

    training = build_dataset(args.seed, args.samples)
    evaluation = build_dataset(args.seed + 1, args.eval_samples)
    weights = train(training, args.epochs, args.learning_rate, args.l2)
    train_acc = accuracy(weights, training)
    eval_acc = accuracy(weights, evaluation)
    matrix = confusion(weights, evaluation)
    write_model(args.out, weights, train_acc, eval_acc, matrix)
    print(
        json.dumps(
            {
                "model": str(args.out),
                "trainAccuracy": round(train_acc, 4),
                "evalAccuracy": round(eval_acc, 4),
                "samples": args.samples,
                "evalSamples": args.eval_samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
