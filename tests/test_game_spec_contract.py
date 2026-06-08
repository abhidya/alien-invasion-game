"""Contract test for recommendation A: one source of truth for game rules.

The physics constants that gate sim-to-real transfer exist in three places --
the canonical ``game_spec.json``, the Python adapter ``tools/game_spec.py``, and
the JS literal ``js/game-spec.js`` consumed by the browser runtime. They MUST
agree: an agent trained against the trainer is deployed into the browser. This
test pins all three together so drift fails CI instead of silently producing a
worse-playing agent.
"""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_JSON = ROOT / "game_spec.json"
SPEC_JS = ROOT / "js" / "game-spec.js"


def _strip_private(node):
    """Drop documentation-only keys (leading underscore) before comparison."""
    if isinstance(node, dict):
        return {k: _strip_private(v) for k, v in node.items() if not k.startswith("_")}
    if isinstance(node, list):
        return [_strip_private(v) for v in node]
    return node


class GameSpecContractTest(unittest.TestCase):
    def setUp(self):
        with open(SPEC_JSON, encoding="utf-8") as handle:
            self.canonical = _strip_private(json.load(handle))

    def test_python_adapter_matches_canonical(self):
        from tools import game_spec

        spec = self.canonical
        self.assertEqual(game_spec.CANVAS_WIDTH, spec["canvas"]["width"])
        self.assertEqual(game_spec.CANVAS_HEIGHT, spec["canvas"]["height"])
        self.assertEqual(game_spec.SHIP_SPEED, spec["ship"]["speed"])
        self.assertEqual(game_spec.SHIP_VERTICAL_SPEED, spec["ship"]["verticalSpeed"])
        self.assertEqual(game_spec.BULLET_SPEED, spec["bullet"]["speed"])
        self.assertEqual(game_spec.ENEMY_SHOT_SPEED, spec["enemyShot"]["speed"])
        self.assertEqual(game_spec.FLEET_DROP, spec["fleet"]["drop"])
        self.assertEqual(game_spec.FLEET_BASE_SPEED, spec["fleet"]["baseSpeed"])
        self.assertEqual(game_spec.FLEET_SPEED_PER_WAVE, spec["fleet"]["speedPerWave"])
        self.assertEqual(game_spec.ACTION_DT, spec["timing"]["actionDt"])
        # Derived values stay consistent with the factor in the spec.
        self.assertAlmostEqual(
            game_spec.ENEMY_SHIP_CONTROL_STEP_Y,
            spec["fleet"]["drop"] * spec["enemyControl"]["stepYFactor"],
        )
        # Fleet shape helpers reproduce the historical formulas.
        self.assertEqual(game_spec.fleet_columns(0), 6)
        self.assertEqual(game_spec.fleet_columns(2), 8)
        self.assertEqual(game_spec.fleet_columns(99), 9)
        self.assertEqual(game_spec.fleet_rows(0), 3)
        self.assertEqual(game_spec.fleet_rows(4), 5)

    @unittest.skipUnless(shutil.which("node"), "node not available to evaluate js/game-spec.js")
    def test_js_literal_matches_canonical(self):
        # Evaluate the real JS file in node so we compare the value the browser
        # actually sees, not a hand-parsed approximation.
        script = (
            "global.window = {};"
            f"require({json.dumps(str(SPEC_JS))});"
            "process.stdout.write(JSON.stringify(global.window.GAME_SPEC));"
        )
        out = subprocess.check_output(["node", "-e", script], text=True)
        js_spec = _strip_private(json.loads(out))
        self.assertEqual(
            js_spec,
            self.canonical,
            "js/game-spec.js drifted from game_spec.json -- regenerate the JS literal.",
        )


if __name__ == "__main__":
    unittest.main()
