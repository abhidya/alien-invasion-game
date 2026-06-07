"""Resume GalagAI training, prune exported models, and publish the static demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = Path(".training-checkpoints/galagai-balanced-v14")
DEFAULT_MODEL = Path("js/galagai-model.json")
STATIC_PAGE_PATHS = [
    Path("index.html"),
    Path("style.css"),
    Path("js"),
    Path("alien_invasion/DQN.py"),
    Path("alien_invasion/images"),
]


def run(command: Sequence[str], *, cwd: Path = ROOT, capture: bool = False) -> str:
    print("+", " ".join(command), flush=True)
    if capture:
        completed = subprocess.run(command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE)
        return completed.stdout
    subprocess.run(command, cwd=cwd, check=True)
    return ""


def checkpoint_rounds(checkpoint_dir: Path) -> int:
    state_path = checkpoint_dir / "state.json"
    if not state_path.exists():
        return 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    rounds = state.get("rounds", [])
    return len(rounds) if isinstance(rounds, list) else int(state.get("roundNumber", 0))


def load_manifest(model_path: Path) -> dict[str, object]:
    return json.loads(model_path.read_text(encoding="utf-8"))


def artifact_summary(model_path: Path) -> dict[str, object]:
    manifest = load_manifest(model_path)
    versions = manifest.get("versions", {})
    if not isinstance(versions, dict):
        raise RuntimeError("Model manifest is missing versions.")
    pilot_versions = versions.get("pilot", [])
    enemy_versions = versions.get("enemies", [])
    if not isinstance(pilot_versions, list) or not isinstance(enemy_versions, list):
        raise RuntimeError("Model manifest versions must be arrays.")
    if not pilot_versions or not enemy_versions:
        raise RuntimeError("Model manifest must contain pilot and enemy versions.")

    model_dir = model_path.parent
    for entry in [*pilot_versions, *enemy_versions]:
        if not isinstance(entry, dict) or not entry.get("url"):
            raise RuntimeError("A model version is missing its URL.")
        checkpoint_path = model_dir / str(entry["url"])
        if not checkpoint_path.exists():
            raise RuntimeError(f"Missing checkpoint JSON: {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        network = checkpoint.get("network", {})
        if not isinstance(network, dict) or not network.get("layers"):
            raise RuntimeError(f"Checkpoint has no exported network: {checkpoint_path}")

    metrics = manifest.get("metrics", {})
    self_play = metrics.get("selfPlay", {}) if isinstance(metrics, dict) else {}
    latest = self_play.get("latest", {}) if isinstance(self_play, dict) else {}
    return {
        "version": manifest.get("version"),
        "pilotVersions": len(pilot_versions),
        "enemyVersions": len(enemy_versions),
        "latestPilot": manifest.get("networkRef"),
        "latestEnemy": manifest.get("enemies", {}).get("networkRef") if isinstance(manifest.get("enemies"), dict) else None,
        "pilotShotAccuracy": metrics.get("pilotShotAccuracy") if isinstance(metrics, dict) else None,
        "waveClearRate": metrics.get("waveClearRate") if isinstance(metrics, dict) else None,
        "latestRound": latest.get("round") if isinstance(latest, dict) else None,
        "latestTrained": latest.get("trained") if isinstance(latest, dict) else None,
        "latestGeneration": latest.get("generation") if isinstance(latest, dict) else None,
    }


def verify_artifacts(model_path: Path, *, skip_tests: bool) -> dict[str, object]:
    summary = artifact_summary(model_path)
    if not skip_tests:
        run([sys.executable, "-m", "py_compile", "tools/train_static_pilot.py", "tools/train_publish.py"])
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
        run(["node", "-c", "js/galagai.js"])
    print(json.dumps({"artifact": summary}, indent=2), flush=True)
    return summary


def build_train_command(
    args: argparse.Namespace,
    target_rounds: int,
    *,
    required_new_balanced_rounds: int = 0,
    force_resume: bool = False,
    force_no_progress: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "tools/train_static_pilot.py",
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--balanced-rounds",
        str(target_rounds),
        "--min-balanced-rounds",
        str(args.min_balanced_rounds),
        "--required-new-balanced-rounds",
        str(max(0, required_new_balanced_rounds)),
        "--balance-tolerance",
        str(args.balance_tolerance),
        "--balance-patience",
        str(args.balance_patience),
        "--balance-min-win-rate",
        str(args.balance_min_win_rate),
        "--phase-timesteps",
        str(args.phase_timesteps),
        "--max-phase-iterations",
        str(args.max_phase_iterations),
        "--eval-episodes",
        str(args.eval_episodes),
        "--max-steps",
        str(args.max_steps),
        "--dominance-threshold",
        str(args.dominance_threshold),
        "--train-workers",
        str(args.train_workers),
        "--eval-workers",
        str(args.eval_workers),
        "--curriculum-waves",
        str(args.curriculum_waves),
        "--candidate-spawns",
        str(args.candidate_spawns),
        "--checkpoint-retention",
        args.checkpoint_retention,
        "--keep-latest-versions",
        str(args.keep_latest_versions),
        "--out",
        str(args.model),
    ]
    if force_resume or ((args.checkpoint_dir / "state.json").exists() and not args.no_resume):
        command.insert(2, "--resume")
    if args.no_progress or force_no_progress:
        command.append("--no-progress")
    return command


def run_training(
    args: argparse.Namespace,
    target_rounds: int,
    current_rounds: int,
    *,
    required_new_balanced_rounds: int | None = None,
) -> bool:
    required_new_balanced_rounds = (
        max(0, target_rounds - current_rounds)
        if required_new_balanced_rounds is None
        else max(0, required_new_balanced_rounds)
    )
    try:
        run(build_train_command(args, target_rounds, required_new_balanced_rounds=required_new_balanced_rounds))
        return False
    except KeyboardInterrupt:
        interrupted = True
    except subprocess.CalledProcessError as error:
        if error.returncode not in {-2, 130}:
            raise
        interrupted = True

    if not interrupted:
        return False

    recovered_rounds = checkpoint_rounds(args.checkpoint_dir)
    if recovered_rounds <= current_rounds:
        raise RuntimeError(
            "Training was interrupted before a new completed generation checkpoint was written. "
            "Run the same train_publish.py command again to resume."
        )
    print(
        json.dumps(
            {
                "trainingInterrupted": True,
                "recoveredCheckpointRounds": recovered_rounds,
                "requestedTargetRounds": target_rounds,
                "action": "exporting and publishing latest completed checkpoint",
            },
            indent=2,
        ),
        flush=True,
    )
    run(
        build_train_command(
            args,
            recovered_rounds,
            required_new_balanced_rounds=0,
            force_resume=True,
            force_no_progress=True,
        )
    )
    return True


def git_has_changes(paths: Sequence[Path] | None = None, *, cached: bool = False, cwd: Path = ROOT) -> bool:
    command = ["git", "diff", "--quiet"]
    if cached:
        command.insert(2, "--cached")
    if paths:
        command.append("--")
        command.extend(str(path) for path in paths)
    return subprocess.run(command, cwd=cwd).returncode != 0


def commit_master(summary: dict[str, object], *, no_push: bool) -> bool:
    run(["git", "add", "--all", "js/galagai-model.json", "js/galagai-models"])
    if not git_has_changes(cached=True):
        print("No master model changes to commit.", flush=True)
        return False

    message = "\n".join(
        [
            "Publish trained GalagAI agents",
            "",
            f"Publish schema {summary['version']} agent JSON with {summary['pilotVersions']} pilot versions and {summary['enemyVersions']} enemy versions.",
            "",
            "Constraint: GitHub Pages can only load checked-in static JSON artifacts.",
            "Rejected: Manual git add/commit/push workflow | it misses gh-pages and lets checkpoint files grow without retention.",
            "Confidence: high",
            "Scope-risk: narrow",
            "Directive: Use tools/train_publish.py for future model pushes so retention and Pages stay synchronized.",
            "Tested: trainer artifact integrity check; py_compile; unittest discovery; node -c js/galagai.js.",
            "Not-tested: Live human gameplay after CDN propagation.",
        ]
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as message_file:
        message_file.write(message)
        message_path = Path(message_file.name)
    try:
        run(["git", "commit", "-F", str(message_path)])
    finally:
        message_path.unlink(missing_ok=True)
    if not no_push:
        run(["git", "push", "origin", "master"])
    return True


def copy_static_pages_files(worktree: Path) -> None:
    for relative_path in STATIC_PAGE_PATHS:
        source = ROOT / relative_path
        if not source.exists():
            continue
        destination = worktree / relative_path
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def publish_pages(summary: dict[str, object], *, no_push: bool) -> bool:
    temp_dir = Path(tempfile.mkdtemp(prefix="galagai-gh-pages."))
    try:
        run(["git", "worktree", "add", str(temp_dir), "origin/gh-pages"])
        copy_static_pages_files(temp_dir)
        run(["git", "add", "--all", "."], cwd=temp_dir)
        if not git_has_changes(cached=True, cwd=temp_dir):
            print("No gh-pages changes to commit.", flush=True)
            return False
        message = "\n".join(
            [
                "Publish trained GalagAI demo agents",
                "",
                f"Mirror schema {summary['version']} static model files to Pages.",
                "",
                "Constraint: Pages serves the gh-pages branch, not master.",
                "Rejected: Push master only | the live demo would keep serving stale models.",
                "Confidence: high",
                "Scope-risk: narrow",
                "Directive: Keep gh-pages synchronized with every accepted model artifact.",
                "Tested: gh-pages worktree manifest/checkpoint integrity through tools/train_publish.py.",
                "Not-tested: CDN propagation timing.",
            ]
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as message_file:
            message_file.write(message)
            message_path = Path(message_file.name)
        try:
            run(["git", "commit", "-F", str(message_path)], cwd=temp_dir)
        finally:
            message_path.unlink(missing_ok=True)
        if not no_push:
            run(["git", "push", "origin", "HEAD:gh-pages"], cwd=temp_dir)
        return True
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(temp_dir)], cwd=ROOT, check=False)
        shutil.rmtree(temp_dir, ignore_errors=True)


def public_manifest_check(expected: dict[str, object], *, attempts: int = 18, delay_seconds: float = 10.0) -> None:
    expected_public = {
        "version": expected["version"],
        "pilotVersions": expected["pilotVersions"],
        "enemyVersions": expected["enemyVersions"],
        "latestPilot": expected["latestPilot"],
        "latestEnemy": expected["latestEnemy"],
    }
    last_summary: dict[str, object] | None = None
    attempts = max(1, int(attempts))
    delay_seconds = max(0.0, float(delay_seconds))
    for attempt in range(1, attempts + 1):
        url = f"https://abhidya.github.io/alien-invasion-game/js/galagai-model.json?model-check={int(time.time())}-{attempt}"
        try:
            output = run(["curl", "-fsSL", "--max-time", "20", url], capture=True)
            data = json.loads(output)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
            print(f"Public manifest check attempt {attempt} skipped/failed: {error}", flush=True)
            if attempt < attempts:
                time.sleep(delay_seconds)
            continue
        public_summary = {
            "version": data.get("version"),
            "pilotVersions": len(data.get("versions", {}).get("pilot", [])),
            "enemyVersions": len(data.get("versions", {}).get("enemies", [])),
            "latestPilot": data.get("networkRef"),
            "latestEnemy": data.get("enemies", {}).get("networkRef") if isinstance(data.get("enemies"), dict) else None,
        }
        last_summary = public_summary
        print(json.dumps({"publicManifest": public_summary, "attempt": attempt}, indent=2), flush=True)
        if public_summary == expected_public:
            return
        if attempt < attempts:
            time.sleep(delay_seconds)
    raise RuntimeError(
        f"Public Pages manifest is not serving the expected model artifact after {attempts} attempts. "
        "The master and gh-pages pushes may already be complete; rerun with --skip-public-check "
        "if GitHub Pages propagation is unusually slow. "
        f"Expected {expected_public}, saw {last_summary}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, prune, commit, push, and publish GalagAI model artifacts.")
    parser.add_argument("--add-rounds", type=int, default=24, help="Train this many more balanced rounds from the current checkpoint.")
    parser.add_argument("--target-rounds", type=int, default=None, help="Absolute balanced round target. Overrides --add-rounds.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--phase-timesteps", type=int, default=900)
    parser.add_argument("--max-phase-iterations", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--dominance-threshold", type=float, default=0.65)
    parser.add_argument("--curriculum-waves", type=int, default=3)
    parser.add_argument("--candidate-spawns", type=int, default=1)
    parser.add_argument("--min-balanced-rounds", type=int, default=12)
    parser.add_argument("--balance-tolerance", type=float, default=0.2)
    parser.add_argument("--balance-patience", type=int, default=3)
    parser.add_argument("--balance-min-win-rate", type=float, default=0.25)
    parser.add_argument("--train-workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--checkpoint-retention", choices=("all", "tiered"), default="tiered")
    parser.add_argument("--keep-latest-versions", type=int, default=12)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-train", action="store_true", help="Only verify/commit/publish current model files.")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--no-pages", action="store_true")
    parser.add_argument("--skip-public-check", action="store_true")
    parser.add_argument("--public-check-attempts", type=int, default=18)
    parser.add_argument("--public-check-delay", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current_rounds = checkpoint_rounds(args.checkpoint_dir)
    target_rounds = args.target_rounds if args.target_rounds is not None else current_rounds + args.add_rounds
    if target_rounds < current_rounds:
        raise RuntimeError(f"Target rounds {target_rounds} is behind current checkpoint rounds {current_rounds}.")

    required_new_balanced_rounds = max(0, target_rounds - current_rounds)
    print(
        json.dumps(
            {
                "currentRounds": current_rounds,
                "targetRounds": target_rounds,
                "requiredNewBalancedRounds": required_new_balanced_rounds,
            },
            indent=2,
        ),
        flush=True,
    )
    interrupted = False
    if not args.no_train:
        interrupted = run_training(
            args,
            target_rounds,
            current_rounds,
            required_new_balanced_rounds=required_new_balanced_rounds,
        )

    summary = verify_artifacts(args.model, skip_tests=args.skip_tests)
    if interrupted:
        print(json.dumps({"publishedInterruptedCheckpoint": summary}, indent=2), flush=True)
    if not args.no_commit:
        commit_master(summary, no_push=args.no_push)
    if not args.no_pages:
        publish_pages(summary, no_push=args.no_push)
    if not args.skip_public_check and not args.no_push and not args.no_pages:
        public_manifest_check(
            summary,
            attempts=args.public_check_attempts,
            delay_seconds=args.public_check_delay,
        )


if __name__ == "__main__":
    main()
