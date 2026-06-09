"""Train every RL family into its own brain manifest and assemble the index.

For each selected algorithm this runs tools/train_publish.py once, into a
per-technique checkpoint directory and output manifest, then writes a ``brains``
index into the main js/galagai-model.json so the browser brain-selector
(js/model-lab.js) can load and play each technique, and mix them per side.

Layout produced:
    js/galagai-model.json          <- DQN (default/live brain) + brains index
    js/brains/<technique>.json     <- one manifest per non-default technique
    js/brains/<technique>-models/  <- that technique's checkpoint networks

Training itself needs the .venv-rl stack (torch + stable-baselines3). This
module's orchestration and index assembly are dependency-free and unit-tested;
the actual ``train_publish`` subprocess is what requires torch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import rl_algorithms

SCHEMA_VERSION = 16
MAIN_MANIFEST = Path("js/galagai-model.json")
BRAINS_DIR = Path("js/brains")
DEFAULT_TECHNIQUES = ["dqn", "qrdqn", "ppo", "a2c", "maskable-ppo"]


def brain_output(algorithm: str) -> Path:
    """Manifest path for an algorithm: the main manifest for DQN, else js/brains/<id>.json."""
    if algorithm == rl_algorithms.DEFAULT_ALGORITHM:
        return MAIN_MANIFEST
    technique = rl_algorithms.get_algorithm(algorithm).technique
    return BRAINS_DIR / f"{technique}.json"


def checkpoint_dir(algorithm: str) -> Path:
    return Path(".training-checkpoints") / rl_algorithms.checkpoint_dir_name(algorithm, SCHEMA_VERSION)


def brain_manifest_url(algorithm: str) -> str:
    """URL of a brain manifest relative to the main manifest's directory (js/)."""
    return brain_output(algorithm).relative_to(MAIN_MANIFEST.parent).as_posix()


def publish_command(
    algorithm: str,
    *,
    target_rounds: int,
    shared_args: list[str],
    python: str = sys.executable,
) -> list[str]:
    return [
        python,
        "tools/train_publish.py",
        "--algorithm",
        algorithm,
        "--checkpoint-dir",
        str(checkpoint_dir(algorithm)),
        "--out",
        str(brain_output(algorithm)),
        "--target-rounds",
        str(target_rounds),
        "--no-push",
        "--no-pages",
        "--no-commit",
        "--skip-tests",
        *shared_args,
    ]


def assemble_brains_index(main_manifest_path: Path, algorithms: list[str]) -> dict[str, object]:
    """Write a ``brains`` index into the main manifest for the non-default techniques.

    The default (DQN) brain is the main manifest itself and stays implicit; only
    techniques with their own manifest file are listed. Returns the index written.
    """
    path = Path(main_manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    brains: dict[str, object] = {}
    for algorithm in algorithms:
        if algorithm == rl_algorithms.DEFAULT_ALGORITHM:
            continue
        spec = rl_algorithms.get_algorithm(algorithm)
        if not brain_output(algorithm).exists():
            continue  # not trained yet; skip
        brains[spec.technique] = {
            "manifest": brain_manifest_url(algorithm),
            "algorithm": spec.manifest_algorithm,
        }
    data["brains"] = brains
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    return brains


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all RL families and assemble the brain index.")
    parser.add_argument("--techniques", nargs="+", default=DEFAULT_TECHNIQUES, choices=rl_algorithms.algorithm_keys())
    parser.add_argument("--target-rounds", type=int, default=4, help="Balanced rounds per technique (a couple gens/side).")
    parser.add_argument("--replay-buffer-size", type=int, default=10_000, help="Smaller buffer keeps grid replay pickles bounded.")
    parser.add_argument("--curriculum-waves", type=int, default=3)
    parser.add_argument("--candidate-spawns", type=int, default=2)
    parser.add_argument("--train-workers", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=4)
    # Speed knobs forwarded to train_publish -> trainer (lower = faster matrix).
    parser.add_argument("--phase-timesteps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--max-phase-iterations", type=int, default=None)
    parser.add_argument("--min-balanced-rounds", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true", help="Start each technique's checkpoint dir fresh.")
    parser.add_argument("--assemble-only", action="store_true", help="Skip training; just (re)write the brains index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shared_args = [
        "--replay-buffer-size", str(args.replay_buffer_size),
        "--curriculum-waves", str(args.curriculum_waves),
        "--candidate-spawns", str(args.candidate_spawns),
        "--train-workers", str(args.train_workers),
        "--eval-workers", str(args.eval_workers),
    ]
    for flag, value in (
        ("--phase-timesteps", args.phase_timesteps),
        ("--max-steps", args.max_steps),
        ("--eval-episodes", args.eval_episodes),
        ("--max-phase-iterations", args.max_phase_iterations),
        ("--min-balanced-rounds", args.min_balanced_rounds),
    ):
        if value is not None:
            shared_args += [flag, str(value)]
    if args.no_resume:
        shared_args.append("--no-resume")

    if not args.assemble_only:
        BRAINS_DIR.mkdir(parents=True, exist_ok=True)
        for algorithm in args.techniques:
            command = publish_command(algorithm, target_rounds=args.target_rounds, shared_args=shared_args)
            print(f"\n=== training {algorithm} -> {brain_output(algorithm)} ===", flush=True)
            print("+ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)

    index = assemble_brains_index(MAIN_MANIFEST, args.techniques)
    print(json.dumps({"brainsIndex": index}, indent=2), flush=True)
    print(
        "\nDone. Review js/galagai-model.json + js/brains/, then publish with:\n"
        "  python tools/train_publish.py --skip-tests --no-resume --assemble-only  # (or your normal push)\n"
        "or push master + gh-pages however you deploy.",
        flush=True,
    )


if __name__ == "__main__":
    main()
