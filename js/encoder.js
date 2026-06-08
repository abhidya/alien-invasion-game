/*
 * GalagAI observation encoder -- single source of truth for the browser runtime.
 *
 * Recommendation B (one encoder behind one interface): the 8x28x48 grid + 6
 * scalar observation is the contract between the headless trainer and the
 * browser. It used to be hand-mirrored in js/galagai.js and
 * tools/train_static_pilot.py with parity asserted only by reading both files.
 *
 * This module is the canonical browser-side encoder. It is pure (no DOM, no
 * globals) and consumes a runtime-neutral "scene": the browser fills a scene
 * from its live state, the trainer's Python encoder produces the same layout,
 * and tests/test_encoder_contract.py + tests/test_encoder_js_parity.py pin both
 * to committed golden vectors. Drift fails CI instead of silently feeding a
 * trained agent a garbled observation.
 *
 * Loadable both as a browser global (window.GalagAIEncoder) and as a Node module
 * (module.exports) so the same code the browser runs is the code the test runs.
 *
 * Scene schema:
 *   {
 *     canvas: { width, height },
 *     ship: { x, y, width, height },
 *     bullets:    [ { x, y, width, height }, ... ],
 *     enemyShots: [ { x, y, width, height }, ... ],
 *     aliens:     [ { x, y, width, height, role, alive }, ... ],  // role in bee|butterfly|boss
 *     controlledIndex: number | null,   // index into aliens, or null for the pilot view
 *     fireReady: boolean,               // pilot fire off cooldown
 *     wave: number,
 *     lives: number,
 *     controlledCanFire: boolean,       // runtime-computed; placed verbatim
 *     controlledRoleValue: number       // runtime-computed; placed verbatim
 *   }
 */
(function (global) {
  "use strict";

  var GRID_CHANNELS = [
    "ship",
    "pilot_bullet",
    "enemy_shot",
    "bee",
    "butterfly",
    "boss",
    "controlled_enemy",
    "danger_lane"
  ];
  var GRID_ROWS = 28;
  var GRID_COLS = 48;
  var ROLE_CHANNEL = { bee: 3, butterfly: 4, boss: 5 };

  function clamp(value, lo, hi) {
    return value < lo ? lo : value > hi ? hi : value;
  }

  function emptyGrid() {
    return new Array(GRID_CHANNELS.length * GRID_ROWS * GRID_COLS).fill(0);
  }

  function markRect(grid, rect, channel, value, width, height) {
    if (!rect) return;
    var x0 = clamp(Math.floor((rect.x / width) * GRID_COLS), 0, GRID_COLS - 1);
    var x1 = clamp(Math.floor(((rect.x + rect.width) / width) * GRID_COLS), 0, GRID_COLS - 1);
    var y0 = clamp(Math.floor((rect.y / height) * GRID_ROWS), 0, GRID_ROWS - 1);
    var y1 = clamp(Math.floor(((rect.y + rect.height) / height) * GRID_ROWS), 0, GRID_ROWS - 1);
    for (var y = y0; y <= y1; y += 1) {
      for (var x = x0; x <= x1; x += 1) {
        grid[channel * GRID_ROWS * GRID_COLS + y * GRID_COLS + x] = value;
      }
    }
  }

  function encodeFrame(scene) {
    var W = scene.canvas.width;
    var H = scene.canvas.height;
    var grid = emptyGrid();

    markRect(grid, scene.ship, 0, 1, W, H);

    (scene.bullets || []).forEach(function (bullet) {
      markRect(grid, bullet, 1, 1, W, H);
    });

    (scene.enemyShots || []).forEach(function (shot) {
      markRect(grid, shot, 2, 1, W, H);
    });

    (scene.aliens || []).forEach(function (alien) {
      if (alien.alive === false) return;
      var channel = ROLE_CHANNEL[alien.role];
      if (channel != null) {
        markRect(grid, alien, channel, 1, W, H);
      }
    });

    if (scene.controlledIndex != null) {
      var controlled = (scene.aliens || [])[scene.controlledIndex];
      if (controlled && controlled.alive !== false) {
        markRect(grid, controlled, 6, 1, W, H);
      }
    }

    // Danger lane: a vertical hazard column from each enemy shot down to the
    // bottom of the screen, centred on the shot but ship-width wide.
    (scene.enemyShots || []).forEach(function (shot) {
      markRect(
        grid,
        {
          x: shot.x - scene.ship.width * 0.45,
          y: shot.y,
          width: scene.ship.width,
          height: Math.max(1, H - shot.y)
        },
        7,
        0.5,
        W,
        H
      );
    });

    return grid;
  }

  function encodeScalars(scene) {
    return [
      scene.fireReady ? 1 : 0,
      clamp(scene.wave / 10, 0, 1),
      clamp(scene.lives / 3, 0, 1),
      clamp(scene.ship.y / scene.canvas.height, 0, 1),
      scene.controlledCanFire ? 1 : 0,
      scene.controlledRoleValue || 0
    ];
  }

  function encodeObservation(scene) {
    return encodeFrame(scene).concat(encodeScalars(scene));
  }

  var api = {
    GRID_CHANNELS: GRID_CHANNELS,
    GRID_ROWS: GRID_ROWS,
    GRID_COLS: GRID_COLS,
    encodeFrame: encodeFrame,
    encodeScalars: encodeScalars,
    encodeObservation: encodeObservation
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  if (global) {
    global.GalagAIEncoder = api;
  }
})(typeof window !== "undefined" ? window : typeof globalThis !== "undefined" ? globalThis : this);
