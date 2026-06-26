"""Shared static-demo publication helpers."""

from __future__ import annotations

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]

STATIC_PAGE_PATHS = (
    Path("index.html"),
    Path("style.css"),
    Path("js"),
    Path("game_spec.json"),
    Path("alien_invasion/DQN.py"),
    Path("alien_invasion/images"),
)


def copy_static_pages_files(worktree: Path, *, root: Path = ROOT) -> None:
    """Mirror the static demo files into a gh-pages worktree."""
    for relative_path in STATIC_PAGE_PATHS:
        source = root / relative_path
        if not source.exists():
            continue
        destination = worktree / relative_path
        if destination.exists():
            shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination) if source.is_dir() else shutil.copy2(source, destination)
