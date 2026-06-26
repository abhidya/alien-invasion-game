"""Resume GalagAI training, prune exported models, and publish the static demo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import rl_algorithms  # noqa: E402  (dependency-free registry)
from tools.static_publish import copy_static_pages_files  # noqa: E402

EXPECTED_MODEL_SCHEMA_VERSION = 17
DEFAULT_CHECKPOINT_DIR = Path(".training-checkpoints/galagai-balanced-v17")
DEFAULT_MODEL = Path("js/galagai-model.json")


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


def checkpoint_schema_version(checkpoint_dir: Path) -> int | None:
    state_path = checkpoint_dir / "state.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    schema = state.get("schemaVersion")
    return int(schema) if schema is not None else None


def validate_resume_checkpoint(args: argparse.Namespace) -> None:
    if args.no_resume:
        return
    schema = checkpoint_schema_version(args.checkpoint_dir)
    if schema is None:
        return
    if schema != EXPECTED_MODEL_SCHEMA_VERSION:
        raise RuntimeError(
            f"Checkpoint schema {schema} at {args.checkpoint_dir} does not match current schema "
            f"{EXPECTED_MODEL_SCHEMA_VERSION}. Use {DEFAULT_CHECKPOINT_DIR} for the current run, "
            "or choose a fresh checkpoint directory with --no-resume."
        )


def checkpoint_generation_counts(checkpoint_dir: Path) -> dict[str, int]:
    state_path = checkpoint_dir / "state.json"
    if not state_path.exists():
        return {"pilot": 0, "enemies": 0}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    counts = state.get("totalGenerationCounts")
    if not isinstance(counts, dict):
        counts = state.get("checkpointCounts", {})
    return {
        "pilot": int(counts.get("pilot", 0)) if isinstance(counts, dict) else 0,
        "enemies": int(counts.get("enemies", 0)) if isinstance(counts, dict) else 0,
    }


def checkpoint_is_publishable(checkpoint_dir: Path) -> bool:
    counts = checkpoint_generation_counts(checkpoint_dir)
    return counts["pilot"] > 0 and counts["enemies"] > 0


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
        "--pilot-warmup-generations",
        str(args.pilot_warmup_generations),
        "--enemy-warmup-generations",
        str(args.enemy_warmup_generations),
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
        "--replay-buffer-size",
        str(args.replay_buffer_size),
        "--algorithm",
        args.algorithm,
        "--device",
        args.device,
        "--out",
        str(args.model),
    ]
    if args.require_cuda:
        command.append("--require-cuda")
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
    if not checkpoint_is_publishable(args.checkpoint_dir):
        counts = checkpoint_generation_counts(args.checkpoint_dir)
        raise RuntimeError(
            "Training was interrupted before both pilot and enemy checkpoints existed, so there is "
            f"nothing publishable yet. Current generations: {counts}. "
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


def stop_process_group(process: subprocess.Popen[object], *, timeout_seconds: float = 180.0) -> int:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        return int(process.wait(timeout=timeout_seconds))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            return int(process.wait(timeout=30))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            return int(process.wait())


def publish_current_artifacts(args: argparse.Namespace, *, interrupted: bool = False) -> dict[str, object]:
    summary = verify_artifacts(args.model, skip_tests=args.skip_tests)
    if interrupted:
        print(json.dumps({"publishedInterruptedCheckpoint": summary}, indent=2), flush=True)
    if not args.no_commit:
        commit_master(summary, model_path=args.model, no_push=args.no_push)
    if not args.no_pages:
        publish_pages(summary, no_push=args.no_push)
    return summary


def export_completed_checkpoint(args: argparse.Namespace, recovered_rounds: int) -> None:
    run(
        build_train_command(
            args,
            recovered_rounds,
            required_new_balanced_rounds=0,
            force_resume=True,
            force_no_progress=True,
        )
    )


def run_training_with_publish_interval(
    args: argparse.Namespace,
    target_rounds: int,
    current_rounds: int,
    *,
    required_new_balanced_rounds: int,
) -> bool:
    interval_seconds = max(1.0, float(args.publish_interval_seconds))
    current_rounds = max(0, int(current_rounds))
    required_new_balanced_rounds = max(0, int(required_new_balanced_rounds))
    while current_rounds < target_rounds:
        command = build_train_command(
            args,
            target_rounds,
            required_new_balanced_rounds=required_new_balanced_rounds,
            force_resume=current_rounds > 0,
        )
        print("+", " ".join(command), flush=True)
        process: subprocess.Popen[object] = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        started_at = time.time()
        publish_due_logged = False
        try:
            while True:
                returncode = process.poll()
                recovered_rounds = checkpoint_rounds(args.checkpoint_dir)
                if returncode is not None:
                    if returncode != 0:
                        if returncode in {-signal.SIGINT, 130} and recovered_rounds > current_rounds:
                            if not checkpoint_is_publishable(args.checkpoint_dir):
                                raise RuntimeError(
                                    "Training stopped before both pilot and enemy checkpoints existed, "
                                    "so the latest checkpoint cannot be published yet."
                                )
                            export_completed_checkpoint(args, recovered_rounds)
                            publish_current_artifacts(args, interrupted=True)
                            return False
                        raise subprocess.CalledProcessError(returncode, command)
                    return True

                elapsed = time.time() - started_at
                if elapsed >= interval_seconds and recovered_rounds > current_rounds:
                    if not checkpoint_is_publishable(args.checkpoint_dir):
                        if not publish_due_logged:
                            print(
                                json.dumps(
                                    {
                                        "publishDue": True,
                                        "action": "waiting for both pilot and enemy checkpoints before publishing",
                                        "checkpointGenerations": checkpoint_generation_counts(args.checkpoint_dir),
                                    },
                                    indent=2,
                                ),
                                flush=True,
                            )
                            publish_due_logged = True
                        time.sleep(5)
                        continue
                    print(
                        json.dumps(
                            {
                                "publishDue": True,
                                "elapsedSeconds": round(elapsed, 2),
                                "previousCheckpointRounds": current_rounds,
                                "recoveredCheckpointRounds": recovered_rounds,
                                "action": "stopping trainer, exporting latest checkpoint, committing, pushing, then resuming",
                            },
                            indent=2,
                        ),
                        flush=True,
                    )
                    returncode = stop_process_group(process)
                    if returncode not in {0, -signal.SIGINT, 130}:
                        raise subprocess.CalledProcessError(returncode, command)
                    recovered_rounds = checkpoint_rounds(args.checkpoint_dir)
                    export_completed_checkpoint(args, recovered_rounds)
                    publish_current_artifacts(args, interrupted=True)
                    current_rounds = recovered_rounds
                    required_new_balanced_rounds = max(0, target_rounds - current_rounds)
                    break

                if elapsed >= interval_seconds and not publish_due_logged:
                    print(
                        json.dumps(
                            {
                                "publishDue": True,
                                "action": "waiting for the next completed generation checkpoint",
                                "previousCheckpointRounds": current_rounds,
                            },
                            indent=2,
                        ),
                        flush=True,
                    )
                    publish_due_logged = True
                time.sleep(5)
        except KeyboardInterrupt:
            returncode = stop_process_group(process)
            recovered_rounds = checkpoint_rounds(args.checkpoint_dir)
            if returncode not in {0, -signal.SIGINT, 130}:
                raise subprocess.CalledProcessError(returncode, command)
            if recovered_rounds <= current_rounds:
                raise RuntimeError(
                    "Training was interrupted before a new completed generation checkpoint was written. "
                    "Run the same train_publish.py command again to resume."
                )
            if not checkpoint_is_publishable(args.checkpoint_dir):
                raise RuntimeError(
                    "Training was interrupted before both pilot and enemy checkpoints existed, so there is "
                    "nothing publishable yet. Run the same train_publish.py command again to resume."
                )
            export_completed_checkpoint(args, recovered_rounds)
            publish_current_artifacts(args, interrupted=True)
            return False
    return True


def git_has_changes(paths: Sequence[Path] | None = None, *, cached: bool = False, cwd: Path = ROOT) -> bool:
    command = ["git", "diff", "--quiet"]
    if cached:
        command.insert(2, "--cached")
    if paths:
        command.append("--")
        command.extend(str(path) for path in paths)
    return subprocess.run(command, cwd=cwd).returncode != 0


def commit_master(summary: dict[str, object], *, model_path: Path, no_push: bool) -> bool:
    run(["git", "add", "--all", str(model_path), str(model_path.parent / "galagai-models")])
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
    parser.add_argument("--pilot-warmup-generations", type=int, default=0, help="Train this many pilot generations before adaptive balanced training.")
    parser.add_argument("--enemy-warmup-generations", type=int, default=0, help="Train this many enemy generations before adaptive balanced training.")
    parser.add_argument("--balance-tolerance", type=float, default=0.2)
    parser.add_argument("--balance-patience", type=int, default=3)
    parser.add_argument("--balance-min-win-rate", type=float, default=0.25)
    parser.add_argument("--train-workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--checkpoint-retention", choices=("all", "tiered"), default="tiered")
    parser.add_argument("--keep-latest-versions", type=int, default=12)
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=50_000,
        help="Off-policy replay buffer capacity. Lower it (e.g. 10000) when the observation "
        "is a large board grid to keep replay pickles from filling the disk.",
    )
    parser.add_argument(
        "--algorithm",
        choices=rl_algorithms.algorithm_keys(),
        default=rl_algorithms.DEFAULT_ALGORITHM,
        help="RL family to train and export (dqn, qrdqn, ppo, a2c, maskable-ppo). "
        "Pair each non-default algorithm with its own --checkpoint-dir and --out.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("GALAGAI_TORCH_DEVICE", "auto"),
        help="Torch/SB3 device selector forwarded to the trainer: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail before training if CUDA is requested but unavailable.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-train", action="store_true", help="Only verify/commit/publish current model files.")
    parser.add_argument(
        "--publish-interval-seconds",
        type=float,
        default=0.0,
        help="When positive, interrupt after this many seconds once a new checkpoint exists, publish it, and resume training.",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--no-pages", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_resume_checkpoint(args)
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
        if args.publish_interval_seconds > 0:
            should_publish_final = run_training_with_publish_interval(
                args,
                target_rounds,
                current_rounds,
                required_new_balanced_rounds=required_new_balanced_rounds,
            )
            if not should_publish_final:
                return
        else:
            interrupted = run_training(
                args,
                target_rounds,
                current_rounds,
                required_new_balanced_rounds=required_new_balanced_rounds,
            )

    publish_current_artifacts(args, interrupted=interrupted)


if __name__ == "__main__":
    main()
