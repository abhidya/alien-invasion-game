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
  enemyShot: { speed: 230, speedPerWave: 20 },
  fleet: {
    drop: 18,
    baseSpeed: 38,
    speedPerWave: 13,
    gapX: 78,
    gapY: 54,
    topY: 74,
    startXOffset: 24,
    columns: { base: 6, perWave: 1, max: 9 },
    rows: { base: 3, perWavePeriod: 2, max: 5 }
  },
  timing: {
    actionDt: 0.12,
    dropCooldown: 1.08,
    enemyShotCooldown: 0.0,
    enemyShipDownCooldown: 0.45,
    enemyShipShotCooldown: 0.65
  },
  enemyControl: { stepX: 32, stepYFactor: 0.7 },
  limits: { maxAliensNormalizer: 45, maxEnemyShotsPerStep: 4 }
};
