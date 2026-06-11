/*
 * Canonical GalagAI game rules for the browser runtime.
 *
 * Recommendation A (single source of truth): this object mirrors the repo-root
 * game_spec.json verbatim. The trainer (tools/train_static_pilot.py via
 * tools/game_spec.py) reads the same JSON, so an agent trained headless plays
 * under identical physics in the browser. tests/test_game_spec_contract.py pins
 * this literal to game_spec.json -- do not hand-edit one without the other.
 *
 * Loaded before js/galagai.js in index.html, so window.GAME_SPEC is available
 * synchronously with no fetch / async ordering risk.
 */
window.GAME_SPEC = {
  schemaVersion: 1,
  canvas: { width: 960, height: 560 },
  ship: {
    width: 64,
    height: 48,
    speed: 470,
    verticalSpeed: 330,
    yOffset: 72,
    minYOffset: 170,
    maxYOffset: 56
  },
  alien: { width: 48, height: 34 },
  bullet: { speed: 620 },
  // speedPerWave 0: enemy shot speed no longer ramps with the wave number.
  enemyShot: { speed: 230, speedPerWave: 0 },
  // speedPerWave 0 + constant columns/rows: the fleet never gets faster or
  // larger with the wave number (perWave 0 holds columns at 6; rows.max ==
  // rows.base holds rows at 3). Difficulty comes only from the trained enemy
  // generation, not scripted per-wave buffs.
  fleet: {
    drop: 18,
    baseSpeed: 38,
    speedPerWave: 0,
    gapX: 78,
    gapY: 54,
    topY: 74,
    startXOffset: 24,
    columns: { base: 6, perWave: 0, max: 9 },
    rows: { base: 3, perWavePeriod: 2, max: 3 }
  },
  // enemyThinkCooldown: the browser enemy decision cadence, pinned to actionDt
  // so the deployed policy decides on the same step it was trained at.
  timing: {
    actionDt: 0.12,
    dropCooldown: 1.08,
    enemyShotCooldown: 0.0,
    enemyShipDownCooldown: 0.45,
    enemyShipShotCooldown: 0.65,
    enemyThinkCooldown: 0.12
  },
  // Per-unit "commit clock": an enemy does not descend until its first
  // committed (non-hold) action arms it; then it drops `step` px every
  // `dropEveryActions` committed actions (step grows by `ramp` each drop).
  // `hold` does not advance the clock, so holding pauses the descent. Per-unit,
  // per-action, reset each wave -- never keyed to the wave number.
  descent: {
    dropEveryActions: 6,
    step: 16,
    ramp: 1.0
  },
  enemyControl: { stepX: 32, stepYFactor: 0.7 },
  limits: { maxAliensNormalizer: 45, maxEnemyShotsPerStep: 4 }
};
