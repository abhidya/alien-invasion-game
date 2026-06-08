"""Generate golden observation vectors for the encoder contract (recommendation B).

Builds a set of representative scenes, encodes each with the *real* Python
trainer encoder (the source of truth, since it produces the agents), and writes
both the runtime-neutral scenes and the resulting observation vectors to
tests/fixtures/encoder_golden.json.

- tests/test_encoder_contract.py replays these scenes through the Python encoder
  and asserts the output still matches -- a regression guard on encode_frame.
- tests/test_encoder_js_parity.py feeds the same neutral scenes to js/encoder.js
  via node and asserts it reproduces the golden vectors -- the cross-language
  parity guard.

Run: ./.venv-rl/bin/python tools/gen_encoder_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.train_static_pilot import Actor, HeadlessGalagai, Shot
from tools.game_spec import CANVAS_HEIGHT, CANVAS_WIDTH

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "encoder_golden.json"


def _rect(x, y, w, h):
    return {"x": float(x), "y": float(y), "width": float(w), "height": float(h)}


# Each scene is runtime-neutral. controlledIndex selects which alien (if any) is
# the "controlled enemy" for the enemy-policy view. Scenes deliberately cover:
# ship placement, pilot bullets, enemy shots + their danger lanes, every alien
# role, a controlled alien, overlapping rects, and off-screen clamping.
SCENES = [
    {
        "name": "pilot_view_ship_and_bullets",
        "ship": _rect(448, 488, 64, 48),
        "bullets": [_rect(470, 300, 6, 18), _rect(120, 120, 6, 18)],
        "enemyShots": [],
        "aliens": [
            {**_rect(100, 80, 48, 34), "role": "bee", "alive": True},
            {**_rect(300, 80, 48, 34), "role": "butterfly", "alive": True},
        ],
        "controlledIndex": None,
        "fireReady": True,
        "wave": 1,
        "lives": 3,
    },
    {
        "name": "enemy_shots_make_danger_lanes",
        "ship": _rect(200, 488, 64, 48),
        "bullets": [],
        "enemyShots": [_rect(210, 150, 6, 16), _rect(800, 60, 6, 16)],
        "aliens": [{**_rect(780, 60, 48, 34), "role": "boss", "alive": True}],
        "controlledIndex": 0,
        "fireReady": False,
        "wave": 4,
        "lives": 2,
    },
    {
        "name": "controlled_alien_and_dead_alien",
        "ship": _rect(448, 488, 64, 48),
        "bullets": [_rect(455, 470, 6, 18)],
        "enemyShots": [_rect(460, 300, 6, 16)],
        "aliens": [
            {**_rect(400, 90, 48, 34), "role": "butterfly", "alive": True},
            {**_rect(470, 90, 48, 34), "role": "bee", "alive": False},
            {**_rect(540, 90, 48, 34), "role": "boss", "alive": True},
        ],
        "controlledIndex": 2,
        "fireReady": True,
        "wave": 6,
        "lives": 1,
    },
    {
        "name": "edge_clamping_offscreen",
        "ship": _rect(-20, 540, 64, 48),
        "bullets": [_rect(958, 10, 6, 18)],
        "enemyShots": [_rect(0, 0, 6, 16)],
        "aliens": [{**_rect(930, 80, 48, 34), "role": "bee", "alive": True}],
        "controlledIndex": None,
        "fireReady": False,
        "wave": 9,
        "lives": 3,
    },
]


def build_env_from_scene(scene: dict) -> tuple[HeadlessGalagai, Actor | None]:
    env = HeadlessGalagai(seed=0)
    s = scene["ship"]
    env.ship = Actor(s["x"], s["y"], s["width"], s["height"])
    env.bullets = [Shot(b["x"], b["y"], b["width"], b["height"], 0.0) for b in scene["bullets"]]
    env.enemy_shots = [Shot(b["x"], b["y"], b["width"], b["height"], 0.0) for b in scene["enemyShots"]]
    env.aliens = [
        Actor(a["x"], a["y"], a["width"], a["height"], alive=a["alive"], role=a["role"])
        for a in scene["aliens"]
    ]
    env.fire_cooldown = 0.0 if scene["fireReady"] else 1.0
    env.wave = scene["wave"]
    env.lives = scene["lives"]
    controlled = None
    if scene["controlledIndex"] is not None:
        controlled = env.aliens[scene["controlledIndex"]]
    return env, controlled


def main() -> None:
    canvas = {"width": CANVAS_WIDTH, "height": CANVAS_HEIGHT}
    scenes_out = []
    observations = []
    for scene in SCENES:
        env, controlled = build_env_from_scene(scene)
        obs = env.observation(controlled).tolist()
        # Fold the runtime-computed scalar inputs into the neutral scene so the JS
        # encoder uses identical scalars and we isolate the encoder-assembly parity.
        neutral = dict(scene)
        neutral["canvas"] = canvas
        neutral["controlledCanFire"] = bool(
            controlled is not None and env.can_enemy_fire(controlled)
        )
        neutral["controlledRoleValue"] = float(
            env.enemy_role_value(controlled.role) if controlled is not None else 0.0
        )
        scenes_out.append(neutral)
        observations.append([round(float(v), 6) for v in obs])

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE, "w", encoding="utf-8") as handle:
        json.dump(
            {"canvas": canvas, "scenes": scenes_out, "observations": observations},
            handle,
            indent=1,
        )
    print(f"wrote {len(SCENES)} golden scenes -> {FIXTURE}")


if __name__ == "__main__":
    main()
