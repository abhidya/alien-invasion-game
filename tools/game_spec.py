"""Python adapter for the canonical GalagAI game rules.

Recommendation A (single source of truth): the physics constants that gate
sim-to-real transfer used to live twice -- hand-mirrored in ``js/galagai.js`` and
``tools/train_static_pilot.py``. They now live once, in ``game_spec.json`` at the
repo root. This module loads that file and re-exports the values as the
``UPPER_CASE`` constants the trainer already expects, so the trainer becomes a
thin adapter over the spec rather than a second copy of it.

``tests/test_game_spec_contract.py`` asserts that this loader, the JS literal in
``js/game-spec.js``, and the trainer all agree with ``game_spec.json`` -- drift is
a failing test, not a silently worse-playing agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPEC_PATH = Path(__file__).resolve().parent.parent / "game_spec.json"


def load_spec(path: Path | None = None) -> dict[str, Any]:
    """Load and return the raw canonical spec dict."""
    with open(path or SPEC_PATH, encoding="utf-8") as handle:
        return json.load(handle)


_SPEC = load_spec()

# --- Derived constants (names match the trainer's historical block) ----------
CANVAS_WIDTH = float(_SPEC["canvas"]["width"])
CANVAS_HEIGHT = float(_SPEC["canvas"]["height"])

SHIP_WIDTH = float(_SPEC["ship"]["width"])
SHIP_HEIGHT = float(_SPEC["ship"]["height"])
SHIP_SPEED = float(_SPEC["ship"]["speed"])
SHIP_VERTICAL_SPEED = float(_SPEC["ship"]["verticalSpeed"])
SHIP_Y = CANVAS_HEIGHT - float(_SPEC["ship"]["yOffset"])
SHIP_MIN_Y = CANVAS_HEIGHT - float(_SPEC["ship"]["minYOffset"])
SHIP_MAX_Y = CANVAS_HEIGHT - float(_SPEC["ship"]["maxYOffset"])

ALIEN_WIDTH = float(_SPEC["alien"]["width"])
ALIEN_HEIGHT = float(_SPEC["alien"]["height"])

BULLET_SPEED = float(_SPEC["bullet"]["speed"])
ENEMY_SHOT_SPEED = float(_SPEC["enemyShot"]["speed"])
ENEMY_SHOT_SPEED_PER_WAVE = float(_SPEC["enemyShot"]["speedPerWave"])

FLEET_DROP = float(_SPEC["fleet"]["drop"])
FLEET_BASE_SPEED = float(_SPEC["fleet"]["baseSpeed"])
FLEET_SPEED_PER_WAVE = float(_SPEC["fleet"]["speedPerWave"])
FLEET_GAP_X = float(_SPEC["fleet"]["gapX"])
FLEET_GAP_Y = float(_SPEC["fleet"]["gapY"])
FLEET_TOP_Y = float(_SPEC["fleet"]["topY"])
FLEET_START_X_OFFSET = float(_SPEC["fleet"]["startXOffset"])
FLEET_COLUMNS_BASE = int(_SPEC["fleet"]["columns"]["base"])
FLEET_COLUMNS_PER_WAVE = int(_SPEC["fleet"]["columns"]["perWave"])
FLEET_COLUMNS_MAX = int(_SPEC["fleet"]["columns"]["max"])
FLEET_ROWS_BASE = int(_SPEC["fleet"]["rows"]["base"])
FLEET_ROWS_PER_WAVE_PERIOD = int(_SPEC["fleet"]["rows"]["perWavePeriod"])
FLEET_ROWS_MAX = int(_SPEC["fleet"]["rows"]["max"])

ACTION_DT = float(_SPEC["timing"]["actionDt"])
DROP_COOLDOWN_SECONDS = float(_SPEC["timing"]["dropCooldown"])
ENEMY_SHOT_COOLDOWN_SECONDS = float(_SPEC["timing"]["enemyShotCooldown"])
ENEMY_SHIP_DOWN_COOLDOWN_SECONDS = float(_SPEC["timing"]["enemyShipDownCooldown"])
ENEMY_SHIP_SHOT_COOLDOWN_SECONDS = float(_SPEC["timing"]["enemyShipShotCooldown"])

ENEMY_SHIP_CONTROL_STEP_X = float(_SPEC["enemyControl"]["stepX"])
ENEMY_SHIP_CONTROL_STEP_Y = FLEET_DROP * float(_SPEC["enemyControl"]["stepYFactor"])

# Per-unit "commit clock": an enemy does not descend until its first committed
# (non-hold) action arms it; after that it drops DESCENT_STEP px every
# DESCENT_DROP_EVERY_ACTIONS committed actions (step grows by DESCENT_RAMP each
# drop). hold does not advance the clock. Per-unit, per-action, reset each wave
# -- never keyed to the wave number. See game_spec.json "descent".
_DESCENT = _SPEC.get("descent", {})
DESCENT_DROP_EVERY_ACTIONS = int(_DESCENT.get("dropEveryActions", 6))
DESCENT_STEP = float(_DESCENT.get("step", 16.0))
DESCENT_RAMP = float(_DESCENT.get("ramp", 1.0))

MAX_ALIENS_NORMALIZER = float(_SPEC["limits"]["maxAliensNormalizer"])
MAX_ENEMY_SHOTS_PER_STEP = int(_SPEC["limits"]["maxEnemyShotsPerStep"])


def fleet_columns(wave: int) -> int:
    return min(FLEET_COLUMNS_MAX, FLEET_COLUMNS_BASE + FLEET_COLUMNS_PER_WAVE * wave)


def fleet_rows(wave: int) -> int:
    return min(FLEET_ROWS_MAX, FLEET_ROWS_BASE + wave // FLEET_ROWS_PER_WAVE_PERIOD)
