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
import shutil
import subprocess
import sys
import tempfile
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
    """Manifest path for an algorithm: the main manifest for DQN, else
    js/brains/<technique>/model.json. Each non-default brain gets its own
    directory so their per-version checkpoint files never collide."""
    if algorithm == rl_algorithms.DEFAULT_ALGORITHM:
        return MAIN_MANIFEST
    technique = rl_algorithms.get_algorithm(algorithm).technique
    return BRAINS_DIR / technique / "model.json"


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
    publish_interval: float | None = None,
    python: str = sys.executable,
) -> list[str]:
    command = [
        python,
        "tools/train_publish.py",
        "--algorithm",
        algorithm,
        "--checkpoint-dir",
        str(checkpoint_dir(algorithm)),
        "--model",
        str(brain_output(algorithm)),
        "--target-rounds",
        str(target_rounds),
        "--skip-tests",
    ]
    if publish_interval:
        # Let train_publish commit/push/mirror this technique's progress on a
        # timer (the same periodic-push you used for single runs). The brain
        # index is still (re)assembled + deployed by train_all at the end.
        command += ["--publish-interval-seconds", str(publish_interval)]
    else:
        command += ["--no-push", "--no-pages", "--no-commit"]
    command += shared_args
    return command


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


# Static files mirrored to gh-pages (matches tools/train_publish.py).
STATIC_PAGE_PATHS = ["index.html", "style.css", "js", "game_spec.json", "alien_invasion/DQN.py", "alien_invasion/images"]


def _git(args: list[str], *, cwd: Path = ROOT, check: bool = True) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=check)


def deploy_artifacts(techniques: list[str]) -> None:
    """Commit + push the trained brains to master, then mirror to gh-pages."""
    _git(["add", "js", "game_spec.json"])
    _git(["commit", "-m", f"Publish v16 grid brains: {', '.join(techniques)}"], check=False)
    for attempt in range(4):
        try:
            _git(["push", "origin", "master"])
            break
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise

    _git(["fetch", "origin", "gh-pages"], check=False)
    worktree = Path(tempfile.mkdtemp(prefix="galagai-ghpages."))
    try:
        _git(["worktree", "add", str(worktree), "origin/gh-pages"])
        for relative in STATIC_PAGE_PATHS:
            source = ROOT / relative
            destination = worktree / relative
            if not source.exists():
                continue
            if destination.exists():
                shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination) if source.is_dir() else shutil.copy2(source, destination)
        _git(["add", "-A"], cwd=worktree)
        _git(["commit", "-m", "Publish v16 grid demo (brains + per-side selector)"], cwd=worktree, check=False)
        for attempt in range(4):
            try:
                _git(["push", "origin", "HEAD:gh-pages"], cwd=worktree)
                break
            except subprocess.CalledProcessError:
                if attempt == 3:
                    raise
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=ROOT, check=False)
        shutil.rmtree(worktree, ignore_errors=True)


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
    parser.add_argument(
        "--publish-interval-seconds",
        type=float,
        default=None,
        help="Periodically commit/push/mirror each technique's progress on a timer "
        "(e.g. 600), instead of only deploying once at the end.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Start each technique's checkpoint dir fresh.")
    parser.add_argument("--assemble-only", action="store_true", help="Skip training; just (re)write the brains index.")
    parser.add_argument("--deploy", action="store_true", help="After assembling, commit + push master and mirror gh-pages.")
    parser.add_argument(
        "--deploy-after-each",
        action="store_true",
        help="Assemble + deploy after EACH technique finishes, so it becomes selectable "
        "on the live site immediately rather than waiting for the whole matrix.",
    )
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
            command = publish_command(
                algorithm,
                target_rounds=args.target_rounds,
                shared_args=shared_args,
                publish_interval=args.publish_interval_seconds,
            )
            print(f"\n=== training {algorithm} -> {brain_output(algorithm)} ===", flush=True)
            print("+ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            if args.deploy_after_each:
                # Reassemble with the techniques finished so far and deploy now, so
                # this one is selectable on the live site immediately.
                index = assemble_brains_index(MAIN_MANIFEST, args.techniques)
                print(json.dumps({"brainsIndex": index}, indent=2), flush=True)
                print(f"=== deploying after {algorithm}: master + gh-pages ===", flush=True)
                deploy_artifacts(args.techniques)

    index = assemble_brains_index(MAIN_MANIFEST, args.techniques)
    print(json.dumps({"brainsIndex": index}, indent=2), flush=True)

    if args.deploy and not args.deploy_after_each:
        print("\n=== deploying: master + gh-pages ===", flush=True)
        deploy_artifacts(args.techniques)
        print("Deployed. Live demo: https://abhidya.github.io/alien-invasion-game/#architectures", flush=True)
    elif args.deploy_after_each:
        print("\nDeployed after each technique. Live demo: "
              "https://abhidya.github.io/alien-invasion-game/#architectures", flush=True)
    else:
        print("\nDone (local). Re-run with --deploy to push master + mirror gh-pages.", flush=True)


if __name__ == "__main__":
    main()
