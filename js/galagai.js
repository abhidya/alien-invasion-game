(function () {
  "use strict";

  // Recommendation A: canonical game rules shared with the headless trainer.
  // Loaded by js/game-spec.js (before this file) and pinned to game_spec.json by
  // tests/test_game_spec_contract.py. An agent trained against the trainer plays
  // under these exact physics in the browser.
  var SPEC = window.GAME_SPEC;
  if (!SPEC) {
    throw new Error("GalagAI: js/game-spec.js must load before js/galagai.js");
  }

  var canvas = document.getElementById("game");
  var ctx = canvas.getContext("2d");
  var scoreNode = document.getElementById("score");
  var waveNode = document.getElementById("wave");
  var livesNode = document.getElementById("lives");
  var bestNode = document.getElementById("best");
  var startButton = document.getElementById("start-button");
  var aiButton = document.getElementById("ai-button");
  var enemyModeButton = document.getElementById("enemy-mode-button");
  var resetButton = document.getElementById("reset-button");
  var checkpointNodes = {
    modelName: document.getElementById("model-name"),
    enemyModelName: document.getElementById("enemy-model-name"),
    evalAccuracy: document.getElementById("eval-accuracy"),
    selfPlayRound: document.getElementById("self-play-round"),
    trainedSide: document.getElementById("trained-side"),
    enemyAction: document.getElementById("enemy-action"),
    pilotWinRate: document.getElementById("pilot-win-rate"),
    enemyPressure: document.getElementById("enemy-pressure")
  };
  var versionNodes = {
    pilotSlider: document.getElementById("pilot-version"),
    pilotLabel: document.getElementById("pilot-version-label"),
    enemySlider: document.getElementById("enemy-version"),
    enemyLabel: document.getElementById("enemy-version-label")
  };

  var assets = {
    ship: loadImage("alien_invasion/images/ship1.png"),
    alienA: loadImage("alien_invasion/images/alien1.png"),
    alienB: loadImage("alien_invasion/images/alien2.png")
  };

  var keys = {};
  var pointerX = null;
  var bestScore = Number(localStorage.getItem("galagai-best") || 0);
  var running = false;
  var pilot = false;
  var enemyMode = "progression";
  var pilotModel = null;
  var enemyModel = null;
  var pilotVersions = [];
  var enemyVersions = [];
  var activePilotVersion = 0;
  var activeEnemyVersion = 0;
  // Per-side "brain" (technique) state. The pilot and the enemy can each run a
  // different exported technique at once (e.g. PPO pilot vs DQN enemy). The main
  // manifest is the default brain; manifest.brains maps other technique ids to
  // their own manifest URLs, loaded lazily on selection.
  var brainManifestObjects = {};
  var brainManifestUrls = {};
  var pilotBrainId = null;
  var enemyBrainId = null;
  var modelManifestUrl = "js/galagai-model.json";
  var modelCache = {};
  var pilotLoadTicket = 0;
  var enemyLoadTicket = 0;
  var enemyAction = "hold";
  var lastTime = 0;
  var fireCooldown = 0;
  var enemyThinkCooldown = 0;
  var enemyShotCooldown = 0;
  var enemyDropCooldown = 0;
  var shake = 0;

  var state = createInitialState();
  loadPilotModel();
  updateHud();
  draw(0);

  function loadImage(src) {
    var image = new Image();
    image.src = src;
    return image;
  }

  // Map a trainer algorithm string to a brain-selector technique id (js/model-lab.js).
  function mapAlgorithmToTechnique(algorithm) {
    var a = String(algorithm || "").toLowerCase();
    if (a.indexOf("maskable") >= 0) return "maskable-ppo";
    if (a.indexOf("qr") >= 0 || a.indexOf("quantile") >= 0) return "qr-dqn";
    if (a.indexOf("ppo") >= 0) return "ppo";
    if (a.indexOf("deepset") >= 0 || a.indexOf("attention") >= 0) return "deepset-attn";
    if (a.indexOf("neat") >= 0 || a.indexOf("neuro") >= 0 || a.indexOf("-es") >= 0) return "neuro-es";
    return "dqn";
  }

  // Tell the brain selector which technique actually drives each side, derived
  // from the loaded manifest. Per-side `technique` overrides the top-level
  // algorithm when the unified exporter records it.
  function publishRuntime(model) {
    model = model || {};
    var base = mapAlgorithmToTechnique(model.algorithm);
    window.GalagAIRuntime = {
      algorithm: model.algorithm || null,
      // Reflect the brain actually driving each side (may differ once mixed).
      pilotTechniqueId: pilotBrainId || (model.pilot && model.pilot.technique) || base,
      enemyTechniqueId: enemyBrainId || (model.enemies && model.enemies.technique) || base
    };
    try {
      window.dispatchEvent(new CustomEvent("galagai:runtime", { detail: window.GalagAIRuntime }));
    } catch (e) {
      /* CustomEvent unsupported -- selector keeps its defaults. */
    }
  }

  // Register the techniques that have exported artifacts. The default technique
  // (this manifest) is always available; manifest.brains adds the rest by URL.
  function registerBrains(model) {
    var base = (model.pilot && model.pilot.technique) || mapAlgorithmToTechnique(model.algorithm);
    brainManifestObjects = {};
    brainManifestUrls = {};
    brainManifestObjects[base] = model;
    pilotBrainId = base;
    enemyBrainId = (model.enemies && model.enemies.technique) || base;
    if (model.brains && typeof model.brains === "object") {
      Object.keys(model.brains).forEach(function (tech) {
        var entry = model.brains[tech];
        var url = typeof entry === "string" ? entry : (entry && entry.manifest);
        if (url && !brainManifestObjects[tech]) brainManifestUrls[tech] = url;
      });
    }
  }

  function availableBrains() {
    var ids = {};
    Object.keys(brainManifestObjects).forEach(function (k) { ids[k] = true; });
    Object.keys(brainManifestUrls).forEach(function (k) { ids[k] = true; });
    return Object.keys(ids);
  }

  // A brain manifest's checkpoint files are stored next to that manifest, but
  // hydrateVersion resolves version urls against the main manifest. Rewrite a
  // loaded brain's version urls to absolute (against the brain's own location)
  // so checkpoints load from the brain's directory, not the main one.
  function absolutizeBrainVersions(manifest, brainBaseUrl) {
    function fix(entry) {
      if (entry && entry.url) entry.url = new URL(entry.url, brainBaseUrl).toString();
      if (entry && entry.networkRef) entry.networkRef = new URL(entry.networkRef, brainBaseUrl).toString();
      return entry;
    }
    var versions = manifest.versions || {};
    (versions.pilot || []).forEach(fix);
    (versions.enemies || []).forEach(fix);
    if (manifest.enemies) fix(manifest.enemies);
    return manifest;
  }

  function loadBrainManifest(technique) {
    if (brainManifestObjects[technique]) return Promise.resolve(brainManifestObjects[technique]);
    var url = brainManifestUrls[technique];
    if (!url) return Promise.reject(new Error("brain not exported: " + technique));
    var absolute = new URL(url, new URL(modelManifestUrl, window.location.href)).toString();
    return fetch(absolute)
      .then(function (response) {
        if (!response.ok) throw new Error("brain manifest unavailable");
        return response.json();
      })
      .then(function (manifest) {
        absolutizeBrainVersions(manifest, absolute);
        brainManifestObjects[technique] = manifest;
        return manifest;
      });
  }

  // Swap the technique driving one side. Pilot and enemy are independent, so
  // selecting a pilot brain never disturbs the enemy and vice-versa.
  function setBrain(side, technique) {
    return loadBrainManifest(technique).then(function (manifest) {
      if (side === "pilot") {
        pilotVersions = extractPilotVersions(manifest);
        activePilotVersion = Math.max(0, pilotVersions.length - 1);
        pilotBrainId = technique;
        configureVersionSlider(versionNodes.pilotSlider, pilotVersions, activePilotVersion);
        loadSelectedVersion("pilot");
      } else {
        enemyVersions = extractEnemyVersions(manifest);
        activeEnemyVersion = Math.max(0, enemyVersions.length - 1);
        enemyBrainId = technique;
        updateEnemyModelForMode();
      }
      publishRuntime(manifest);
      updateHud();
      return technique;
    });
  }

  window.GalagAI = {
    setBrain: setBrain,
    availableBrains: availableBrains,
    currentBrain: function (side) { return side === "pilot" ? pilotBrainId : enemyBrainId; }
  };

  function loadPilotModel() {
    fetch(modelManifestUrl)
      .then(function (response) {
        if (!response.ok) throw new Error("model unavailable");
        return response.json();
      })
      .then(function (model) {
        if (!isUsableManifest(model)) return;
        registerBrains(model);
        pilotVersions = extractPilotVersions(model);
        enemyVersions = extractEnemyVersions(model);
        activePilotVersion = Math.max(0, pilotVersions.length - 1);
        activeEnemyVersion = Math.max(0, enemyVersions.length - 1);
        applySelectedVersions();
        publishRuntime(model);
        updateHud();
      })
      .catch(function () {
        pilotModel = null;
        enemyModel = null;
        pilotVersions = [];
        enemyVersions = [];
        updateHud();
      });
  }

  function isUsableManifest(model) {
    return Boolean(
      model &&
      Array.isArray(model.actions) &&
      (isUsableModel(model) || (model.versions && Array.isArray(model.versions.pilot)))
    );
  }

  function isUsableModel(model) {
    return Boolean(
      model &&
      Array.isArray(model.actions) &&
      (Array.isArray(model.weights) || (model.network && Array.isArray(model.network.layers)))
    );
  }

  function isLoadableVersion(model) {
    return Boolean(
      model &&
      Array.isArray(model.actions) &&
      (isUsableModel(model) || model.url || model.networkRef)
    );
  }

  // Per-version checkpoint entries may not carry the inference head fields, so
  // inherit them from the manifest (or per-side block). Keeps quantile folding
  // and action masking working regardless of where the field is recorded.
  function stampHead(versions, head, masking) {
    return versions.map(function (version) {
      return Object.assign({}, version, {
        outputHead: version.outputHead || head,
        actionMasking: version.actionMasking != null ? version.actionMasking : Boolean(masking)
      });
    });
  }

  function extractPilotVersions(model) {
    var versions = model.versions && Array.isArray(model.versions.pilot)
      ? model.versions.pilot.filter(isLoadableVersion)
      : [];
    if (versions.length) return stampHead(versions, model.outputHead, model.actionMasking);
    return stampHead([model], model.outputHead, model.actionMasking);
  }

  function extractEnemyVersions(model) {
    var enemy = model.enemies || {};
    var head = enemy.outputHead || model.outputHead;
    var masking = enemy.actionMasking != null ? enemy.actionMasking : model.actionMasking;
    var versions = model.versions && Array.isArray(model.versions.enemies)
      ? model.versions.enemies.filter(isLoadableVersion)
      : [];
    if (versions.length) return stampHead(versions, head, masking);
    return isUsableModel(model.enemies) ? stampHead([model.enemies], head, masking) : [];
  }

  function applySelectedVersions() {
    configureVersionSlider(versionNodes.pilotSlider, pilotVersions, activePilotVersion);
    configureVersionSlider(versionNodes.enemySlider, enemyVersions, activeEnemyVersion);
    loadSelectedVersion("pilot");
    updateEnemyModelForMode();
  }

  function configureVersionSlider(slider, versions, activeIndex) {
    if (!slider) return;
    slider.min = versions.length ? "1" : "0";
    slider.max = String(Math.max(versions.length, 1));
    slider.value = versions.length ? String(activeIndex + 1) : "0";
    slider.disabled = versions.length <= 1;
  }

  function selectVersion(kind, value) {
    var index = Math.max(0, Number(value || 1) - 1);
    if (kind === "pilot") {
      activePilotVersion = Math.min(index, Math.max(0, pilotVersions.length - 1));
    } else {
      activeEnemyVersion = Math.min(index, Math.max(0, enemyVersions.length - 1));
      enemyMode = "manual";
    }
    applySelectedVersions();
    updateHud();
  }

  function updateEnemyModelForMode() {
    if (enemyMode === "progression") {
      var progressionIndex = enemyIndexForWave(state.wave);
      if (progressionIndex < 0) {
        enemyModel = null;
        configureVersionSlider(versionNodes.enemySlider, enemyVersions, activeEnemyVersion);
        return;
      }
      activeEnemyVersion = progressionIndex;
    }
    configureVersionSlider(versionNodes.enemySlider, enemyVersions, activeEnemyVersion);
    loadSelectedVersion("enemies");
  }

  function enemyIndexForWave(wave) {
    if (!enemyVersions.length) return -1;
    if (enemyVersions.length === 1) return 0;
    var progress = clamp((wave - 1) / 8, 0, 1);
    return Math.min(enemyVersions.length - 1, Math.round(progress * (enemyVersions.length - 1)));
  }

  function loadSelectedVersion(kind) {
    var versions = kind === "pilot" ? pilotVersions : enemyVersions;
    var activeIndex = kind === "pilot" ? activePilotVersion : activeEnemyVersion;
    var entry = versions[activeIndex] || null;
    var ticket;
    if (kind === "pilot") {
      ticket = pilotLoadTicket + 1;
      pilotLoadTicket = ticket;
      pilotModel = isUsableModel(entry) ? entry : null;
    } else {
      ticket = enemyLoadTicket + 1;
      enemyLoadTicket = ticket;
      enemyModel = isUsableModel(entry) ? entry : null;
    }
    if (!entry || isUsableModel(entry)) return;

    hydrateVersion(entry)
      .then(function (model) {
        if (kind === "pilot" && ticket === pilotLoadTicket && activeIndex === activePilotVersion) {
          pilotModel = model;
        } else if (kind === "enemies" && ticket === enemyLoadTicket && activeIndex === activeEnemyVersion) {
          enemyModel = model;
        }
        updateHud();
      })
      .catch(function () {
        if (kind === "pilot" && ticket === pilotLoadTicket) pilotModel = null;
        if (kind === "enemies" && ticket === enemyLoadTicket) enemyModel = null;
        updateHud();
      });
  }

  function hydrateVersion(entry) {
    var url = entry.url || entry.networkRef;
    if (!url) return Promise.resolve(entry);
    var absoluteUrl = new URL(url, new URL(modelManifestUrl, window.location.href)).toString();
    if (!modelCache[absoluteUrl]) {
      modelCache[absoluteUrl] = fetch(absoluteUrl)
        .then(function (response) {
          if (!response.ok) throw new Error("checkpoint unavailable");
          return response.json();
        })
        .then(function (model) {
          return Object.assign({}, entry, model);
        });
    }
    return modelCache[absoluteUrl];
  }

  function createInitialState() {
    return {
      score: 0,
      wave: 1,
      lives: 3,
      message: "Press Start",
      ship: {
        x: canvas.width / 2 - SPEC.ship.width / 2,
        y: canvas.height - SPEC.ship.yOffset,
        width: SPEC.ship.width,
        height: SPEC.ship.height,
        speed: SPEC.ship.speed,
        verticalSpeed: SPEC.ship.verticalSpeed
      },
      bullets: [],
      enemyShots: [],
      aliens: createFleet(1),
      fleetDirection: 1,
      fleetDrop: SPEC.fleet.drop,
      fleetSpeed: SPEC.fleet.baseSpeed
    };
  }

  function createFleet(wave) {
    var aliens = [];
    var columns = Math.min(SPEC.fleet.columns.max, SPEC.fleet.columns.base + SPEC.fleet.columns.perWave * wave);
    var rows = Math.min(SPEC.fleet.rows.max, SPEC.fleet.rows.base + Math.floor(wave / SPEC.fleet.rows.perWavePeriod));
    var gapX = SPEC.fleet.gapX;
    var gapY = SPEC.fleet.gapY;
    var startX = (canvas.width - (columns - 1) * gapX) / 2 - SPEC.fleet.startXOffset;
    for (var row = 0; row < rows; row += 1) {
      for (var col = 0; col < columns; col += 1) {
        aliens.push({
          x: startX + col * gapX,
          y: SPEC.fleet.topY + row * gapY,
          width: SPEC.alien.width,
          height: SPEC.alien.height,
          type: (row + col) % 2,
          role: enemyRoleForSlot(row, col, wave),
          alive: true,
          wobble: Math.random() * Math.PI * 2,
          homeX: startX + col * gapX,
          homeY: 74 + row * gapY,
          free: false,
          loop: false,
          scatter: 0,
          shotCooldown: 0,
          downCooldown: 0
        });
      }
    }
    return aliens;
  }

  function enemyRoleForSlot(row, col, wave) {
    if (wave >= 3 && row === 0 && col % 3 === 1) return "boss";
    if (wave >= 2 && row <= 1 && col % 2 === 0) return "butterfly";
    return "bee";
  }

  function startGame() {
    state = createInitialState();
    running = true;
    lastTime = performance.now();
    state.message = "";
    updateEnemyModelForMode();
    updateHud();
  }

  function endGame(message) {
    running = false;
    state.message = message;
    if (state.score > bestScore) {
      bestScore = state.score;
      localStorage.setItem("galagai-best", String(bestScore));
    }
    updateHud();
  }

  function updateHud() {
    scoreNode.textContent = state.score;
    waveNode.textContent = state.wave;
    livesNode.textContent = state.lives;
    bestNode.textContent = bestScore;
    if (!pilot) {
      aiButton.textContent = "Pilot: manual";
    } else {
      aiButton.textContent = pilotModel ? "Pilot: trained" : "Pilot: heuristic";
    }
    aiButton.setAttribute("aria-pressed", pilot ? "true" : "false");
    if (enemyModeButton) {
      enemyModeButton.textContent = enemyMode === "progression" ? "Enemies: progression" : "Enemies: slider";
      enemyModeButton.setAttribute("aria-pressed", enemyMode === "progression" ? "true" : "false");
    }
    updateCheckpointPanel();
  }

  function updateCheckpointPanel() {
    var pilotEntry = pilotVersions[activePilotVersion] || null;
    var enemyEntry = enemyVersions[activeEnemyVersion] || null;
    var metrics = modelMetrics(pilotModel || pilotEntry);
    var selfPlay = metrics && metrics.selfPlay ? metrics.selfPlay : null;
    var latest = selfPlay && selfPlay.latest ? selfPlay.latest : null;

    checkpointNodes.modelName.textContent = pilotEntry ? modelLabel(pilotModel || pilotEntry) : "heuristic";
    checkpointNodes.enemyModelName.textContent = enemyDisplayLabel(enemyModel || enemyEntry);
    checkpointNodes.evalAccuracy.textContent = metrics ? percent(metrics.evalAccuracy) : "--";
    checkpointNodes.selfPlayRound.textContent = latest ? latest.round : "--";
    checkpointNodes.trainedSide.textContent = latest ? latest.trained : "--";
    checkpointNodes.enemyAction.textContent = enemyAction || "--";
    checkpointNodes.pilotWinRate.textContent = latest ? percent(latest.enemyWinRate) : "--";
    checkpointNodes.enemyPressure.textContent = latest ? percent(latest.enemyDropRate) : "--";
    if (versionNodes.pilotLabel) {
      versionNodes.pilotLabel.textContent = pilotEntry ? versionLabel(pilotModel || pilotEntry, activePilotVersion, pilotVersions.length) : "none";
    }
    if (versionNodes.enemyLabel) {
      versionNodes.enemyLabel.textContent = enemyDisplayLabel(enemyModel || enemyEntry);
    }
  }

  function modelMetrics(model) {
    if (!model) return null;
    if (model.metrics) return model.metrics;
    return {
      evalAccuracy: Number(model.pilotWinRate || 0),
      enemyWinRate: Number(model.enemyWinRate || 0),
      enemyDropRate: Number(model.enemyDropRate || 0),
      invalidDropRate: Number(model.invalidDropRate || 0),
      enemyFireRate: Number(model.enemyFireRate || 0),
      selfPlay: { latest: model }
    };
  }

  function modelLabel(model) {
    return model.label || model.model || "checkpoint";
  }

  function versionLabel(model, activeIndex, count) {
    return modelLabel(model) + " (" + (activeIndex + 1) + "/" + Math.max(count, 1) + ")";
  }

  function enemyDisplayLabel(model) {
    if (!model) return enemyMode === "progression" ? "progression loading" : "scripted";
    return enemyMode === "progression"
      ? "Progression " + versionLabel(model, activeEnemyVersion, enemyVersions.length)
      : versionLabel(model, activeEnemyVersion, enemyVersions.length);
  }

  function percent(value) {
    return Math.round(Number(value || 0) * 100) + "%";
  }

  function loop(now) {
    var dt = Math.min((now - lastTime) / 1000, 0.032);
    lastTime = now;
    if (running) update(dt);
    draw(now / 1000);
    requestAnimationFrame(loop);
  }

  function update(dt) {
    fireCooldown = Math.max(0, fireCooldown - dt);
    enemyThinkCooldown = Math.max(0, enemyThinkCooldown - dt);
    enemyShotCooldown = Math.max(0, enemyShotCooldown - dt);
    enemyDropCooldown = Math.max(0, enemyDropCooldown - dt);
    shake = Math.max(0, shake - dt);
    if (pilot) runPilot(dt);
    updateShip(dt);
    updateBullets(dt);
    runEnemyModel(dt);
    updateAliens(dt);
    maybeAlienFire(dt);
    checkHits();
    if (!state.aliens.some(function (alien) { return alien.alive; })) {
      state.wave += 1;
      state.aliens = createFleet(state.wave);
      state.fleetDirection = 1;
      state.fleetSpeed += SPEC.fleet.speedPerWave;
      state.score += 250;
      updateEnemyModelForMode();
      updateHud();
    }
  }

  function runPilot() {
    if (pilotModel) {
      runModelPilot();
      return;
    }
    var target = nearestThreat();
    if (!target) return;
    var shipCenter = state.ship.x + state.ship.width / 2;
    var targetCenter = target.x + target.width / 2;
    keys.ArrowLeft = targetCenter < shipCenter - 14;
    keys.ArrowRight = targetCenter > shipCenter + 14;
    if (Math.abs(targetCenter - shipCenter) < 44) fire();
  }

  function runModelPilot() {
    var features = modelUsesGrid(pilotModel)
      ? fullGridFeatures(null)
      : pilotFeatures(modelUsesWrap(pilotModel));
    var mask = pilotModel.actionMasking ? pilotActionMask(pilotModel) : null;
    var action = predictModelAction(pilotModel, "stay", features, mask);
    keys.ArrowLeft = action === "left";
    keys.ArrowRight = action === "right";
    keys.ArrowUp = action === "up";
    keys.ArrowDown = action === "down";
    if (action === "fire") fire();
  }

  function predictModelAction(model, fallback, featureOverride, mask) {
    if (!model) return fallback;
    var features = modelFeatures(model, featureOverride);
    var scores = model.network ? evaluateNetwork(model.network, features) : null;
    // QR-DQN emits |A|x N quantiles; the Q-value per action is the mean over
    // quantiles. Fold before argmax when the manifest marks a quantile head.
    if (scores && model.outputHead === "quantiles") {
      scores = foldQuantiles(scores, model.actions.length);
    }
    var bestAction = fallback;
    var bestScoreForAction = -Infinity;
    model.actions.forEach(function (action, actionIndex) {
      // MaskablePPO: skip actions the mask marks illegal (equivalent to setting
      // their logits to -Infinity before the argmax).
      if (mask && mask[actionIndex] === false) return;
      var score = scores
        ? Number(scores[actionIndex] || 0)
        : features.reduce(function (total, value, featureIndex) {
          return total + value * Number(model.weights[featureIndex][actionIndex] || 0);
        }, 0);
      if (score > bestScoreForAction) {
        bestScoreForAction = score;
        bestAction = action;
      }
    });
    return bestAction;
  }

  // Mean-over-quantiles fold for a QR-DQN output. The flat vector is ordered
  // [q0a0, q0a1, ..., q1a0, ...] (n_quantiles x num_actions, row-major), so the
  // quantiles for action a are at stride numActions starting at a.
  function foldQuantiles(scores, numActions) {
    if (!numActions || scores.length % numActions !== 0) return scores;
    var quantiles = scores.length / numActions;
    var folded = new Array(numActions);
    for (var a = 0; a < numActions; a += 1) {
      var sum = 0;
      for (var q = 0; q < quantiles; q += 1) sum += Number(scores[a + q * numActions] || 0);
      folded[a] = sum / quantiles;
    }
    return folded;
  }

  // Legal-action masks matching the trainer's env (action_masks). Only used when
  // the manifest sets actionMasking (MaskablePPO); other models pass no mask.
  function pilotActionMask(model) {
    return model.actions.map(function (action) {
      if (action === "fire") return fireCooldown <= 0;
      return true;
    });
  }

  function enemyActionMask(model, alien) {
    var canFire = canAlienFire(alien);
    var canDown = (alien.downCooldown || 0) <= 0 && (alien.y + alien.height) < canvas.height;
    return model.actions.map(function (action) {
      if (action.indexOf("fire") !== -1 && !canFire) return false;
      if ((action === "down" || action === "down_fire") && !canDown) return false;
      return true;
    });
  }

  function modelFeatures(model, featureOverride) {
    var features;
    if (modelUsesGrid(model)) {
      features = featureOverride || fullGridFeatures(null);
    } else {
      features = featureOverride || pilotFeatures();
    }
    var expected = Array.isArray(model.features) ? model.features.length : features.length;
    return features.slice(0, expected);
  }

  function evaluateNetwork(network, features) {
    var values = features.slice();
    network.layers.forEach(function (layer, layerIndex) {
      var next = layer.biases.map(function (bias, outputIndex) {
        var total = Number(bias || 0);
        values.forEach(function (value, inputIndex) {
          var row = layer.weights[inputIndex] || [];
          total += value * Number(row[outputIndex] || 0);
        });
        if (layerIndex < network.layers.length - 1 && network.activation === "relu") {
          return Math.max(0, total);
        }
        return total;
      });
      values = next;
    });
    return values;
  }

  function runEnemyModel() {
    if (!enemyModel || enemyThinkCooldown > 0 || !running) return;
    var liveAliens = state.aliens.filter(function (alien) { return alien.alive; });
    if (!liveAliens.length) return;

    var summary = { left: 0, right: 0, down: 0, fire: 0, invalid: 0 };
    var wrap = modelUsesWrap(enemyModel);
    liveAliens.forEach(function (alien) {
      var action = predictModelAction(
        enemyModel,
        "hold",
        modelUsesGrid(enemyModel) ? fullGridFeatures(alien) : enemyFeatures(alien, wrap),
        enemyModel.actionMasking ? enemyActionMask(enemyModel, alien) : null
      );
      applyEnemyShipAction(action, alien, summary);
    });
    enemyAction = enemyActionSummary(summary);
    enemyThinkCooldown = Math.max(0.08, 0.20 - state.wave * 0.01);
    updateHud();
  }

  function applyEnemyShipAction(action, alien, summary) {
    if (!alien || !alien.alive) return;
    action = normalizeEnemyShipAction(action);
    if (action === "left" || action === "left_fire") {
      alien.x -= 32;
      wrapAlienHorizontal(alien);
      summary.left += 1;
    } else if (action === "right" || action === "right_fire") {
      alien.x += 32;
      wrapAlienHorizontal(alien);
      summary.right += 1;
    }
    if (action === "down" || action === "down_fire") {
      if ((alien.downCooldown || 0) <= 0) {
        alien.y += state.fleetDrop * SPEC.enemyControl.stepYFactor;
        alien.downCooldown = SPEC.timing.enemyShipDownCooldown;
        summary.down += 1;
      } else {
        summary.invalid += 1;
      }
    }
    if (action.indexOf("fire") !== -1) {
      if (summary.fire < 4 && fireAlienShot(alien)) {
        summary.fire += 1;
      } else {
        summary.invalid += 1;
      }
    }
  }

  function normalizeEnemyShipAction(action) {
    if (action === "drift_left") return "left";
    if (action === "drift_right") return "right";
    if (action === "drift_left_fire") return "left_fire";
    if (action === "drift_right_fire") return "right_fire";
    if (action === "drop" || action === "dive") return "down";
    if (action === "loop") return "down_fire";
    if (action === "scatter") return "left";
    return action || "hold";
  }

  function enemyActionSummary(summary) {
    var parts = [];
    if (summary.left) parts.push(summary.left + " left");
    if (summary.right) parts.push(summary.right + " right");
    if (summary.down) parts.push(summary.down + " down");
    if (summary.fire) parts.push(summary.fire + " fire");
    if (!parts.length && summary.invalid) return "blocked";
    return parts.length ? parts.join(", ") : "hold";
  }

  function fireRoleShot(roles) {
    var shooter = nearestAlienByRole(roles);
    if (!shooter) return false;
    fireAlienShot(shooter);
    return true;
  }

  function launchRoleDive(roles) {
    var diver = nearestAlienByRole(roles);
    if (!diver) return false;
    diver.free = true;
    diver.loop = false;
    diver.scatter = 0;
    return true;
  }

  function startRoleLoop(roles) {
    var looper = nearestAlienByRole(roles);
    if (!looper) return false;
    looper.loop = true;
    looper.free = false;
    looper.scatter = 0;
    return true;
  }

  function scatterRoles(roles) {
    var scattered = false;
    state.aliens.forEach(function (alien, index) {
      if (!alien.alive || roles.indexOf(alien.role) === -1) return;
      alien.scatter = index % 2 === 0 ? -1 : 1;
      alien.free = true;
      alien.loop = false;
      scattered = true;
    });
    return scattered;
  }

  function nearestAlienByRole(roles) {
    var shipCenter = state.ship.x + state.ship.width / 2;
    return state.aliens
      .filter(function (alien) { return alien.alive && roles.indexOf(alien.role) !== -1; })
      .sort(function (a, b) {
        return Math.abs((a.x + a.width / 2) - shipCenter) - Math.abs((b.x + b.width / 2) - shipCenter);
      })[0] || null;
  }

  function modelUsesWrap(model) {
    return Boolean(model && (model.featureEncoding === "wrap-x" ||
      (typeof model.version === "number" && model.version >= 15 && model.featureEncoding !== "grid-v1")));
  }

  function modelUsesGrid(model) {
    return Boolean(model && model.featureEncoding === "grid-v1");
  }

  function relativeX(delta, wrap) {
    // Normalized horizontal delta. When wrap is set, take the shortest signed
    // distance across the toroidal x-axis (matches the Python trainer's
    // _relative_x) so the policy can perceive a screen edge as an escape route.
    var width = canvas.width;
    var half = width / 2;
    if (wrap) {
      delta = (((delta + half) % width) + width) % width - half;
    }
    return clamp(delta / half, -1, 1);
  }

  // Recommendation B: the grid observation is produced by the shared, DOM-free
  // encoder in js/encoder.js (window.GalagAIEncoder) -- the exact layout the
  // Python trainer mirrors and tests/test_encoder_*.py pin to golden vectors. We
  // assemble a runtime-neutral "scene" from live state and hand it to the encoder
  // so the browser and the trainer can never silently disagree on what a state
  // looks like to the agent.
  if (!window.GalagAIEncoder) {
    throw new Error("GalagAI: js/encoder.js must load before js/galagai.js");
  }

  function buildScene(controlledAlien) {
    var controlledIndex = controlledAlien ? state.aliens.indexOf(controlledAlien) : -1;
    return {
      canvas: { width: canvas.width, height: canvas.height },
      ship: state.ship,
      bullets: state.bullets,
      enemyShots: state.enemyShots,
      aliens: state.aliens,
      controlledIndex: controlledIndex >= 0 ? controlledIndex : null,
      fireReady: fireCooldown <= 0,
      wave: state.wave,
      lives: state.lives,
      controlledCanFire: controlledAlien ? canAlienFire(controlledAlien) : false,
      controlledRoleValue: controlledAlien ? enemyRoleValue(controlledAlien.role) : 0
    };
  }

  function fullGridFeatures(controlledAlien) {
    return window.GalagAIEncoder.encodeObservation(buildScene(controlledAlien));
  }

  function pilotFeatures(wrap) {
    var shipCenter = state.ship.x + state.ship.width / 2;
    var liveAliens = state.aliens.filter(function (alien) { return alien.alive; });
    var target = nearestThreat();
    var targetDx = 0;
    var targetDy = 0;
    var alienCount = 0;
    if (target) {
      targetDx = relativeX((target.x + target.width / 2) - shipCenter, wrap);
      targetDy = clamp(
        ((target.y + target.height / 2) - (state.ship.y + state.ship.height / 2)) / canvas.height,
        -1,
        1
      );
      alienCount = liveAliens.length / 45;
    }

    var closestShot = state.enemyShots.slice().sort(function (a, b) {
      return Math.abs((a.x + a.width / 2) - shipCenter) - Math.abs((b.x + b.width / 2) - shipCenter);
    })[0];
    var dangerShot = dangerousEnemyShot(shipCenter);
    var pilotBulletThreat = dangerousPilotBullet(liveAliens);
    var threatDx = closestShot
      ? relativeX((closestShot.x + closestShot.width / 2) - shipCenter, wrap)
      : 1;
    var threatY = closestShot ? clamp(closestShot.y / canvas.height, 0, 1) : 0;
    var dangerDx = dangerShot
      ? relativeX((dangerShot.x + dangerShot.width / 2) - shipCenter, wrap)
      : 1;
    var dangerY = dangerShot ? clamp(dangerShot.y / canvas.height, 0, 1) : 0;
    var dangerLane = dangerShot ? shotLaneOverlap(dangerShot, state.ship) : 0;
    var pilotBulletDx = 0;
    var pilotBulletY = 0;
    if (pilotBulletThreat) {
      pilotBulletDx = relativeX(
        (pilotBulletThreat.bullet.x + pilotBulletThreat.bullet.width / 2) -
          (pilotBulletThreat.alien.x + pilotBulletThreat.alien.width / 2),
        wrap
      );
      pilotBulletY = clamp(
        (pilotBulletThreat.bullet.y - (pilotBulletThreat.alien.y + pilotBulletThreat.alien.height)) / canvas.height,
        -1,
        1
      );
    }
    var beeCount = liveAliens.filter(function (alien) { return alien.role === "bee"; }).length / 45;
    var gunshipCount = liveAliens.filter(function (alien) { return alien.role === "butterfly" || alien.role === "boss"; }).length / 45;
    var bossCount = liveAliens.filter(function (alien) { return alien.role === "boss"; }).length / 45;

    return [
      targetDx,
      Math.abs(targetDx),
      threatDx,
      threatY,
      fireCooldown <= 0 ? 1 : 0,
      clamp(alienCount, 0, 1),
      clamp(state.wave / 10, 0, 1),
      enemyDropCooldown <= 0 ? 1 : 0,
      1,
      fleetProgress(),
      clamp(state.lives / 3, 0, 1),
      dangerDx,
      dangerY,
      dangerLane,
      pilotBulletDx,
      pilotBulletY,
      clamp(beeCount, 0, 1),
      clamp(gunshipCount, 0, 1),
      clamp(bossCount, 0, 1),
      clamp(state.ship.y / canvas.height, 0, 1),
      targetDy
    ];
  }

  function enemyFeatures(alien, wrap) {
    var features = pilotFeatures(wrap);
    var shipCenter = state.ship.x + state.ship.width / 2;
    features.push(relativeX((alien.x + alien.width / 2) - shipCenter, wrap));
    features.push(clamp(alien.y / canvas.height, 0, 1));
    features.push(enemyRoleValue(alien.role));
    features.push(canAlienFire(alien) ? 1 : 0);
    features.push(pilotBulletLaneFor(alien));
    features.push(clamp((alien.y + alien.height) / canvas.height, 0, 1));
    return features;
  }

  function enemyRoleValue(role) {
    if (role === "boss") return 1;
    if (role === "butterfly") return 0.5;
    return 0;
  }

  function canAlienFire(alien) {
    return alien && alien.alive &&
      (alien.role === "butterfly" || alien.role === "boss") &&
      (alien.shotCooldown || 0) <= 0;
  }

  function pilotBulletLaneFor(alien) {
    var best = 0;
    state.bullets.forEach(function (bullet) {
      best = Math.max(best, shotLaneOverlap(bullet, alien));
    });
    return best;
  }

  function dangerousEnemyShot(shipCenter) {
    return state.enemyShots.slice().sort(function (a, b) {
      return enemyShotDangerScore(b, shipCenter) - enemyShotDangerScore(a, shipCenter);
    })[0] || null;
  }

  function enemyShotDangerScore(shot, shipCenter) {
    var laneWidth = Math.max(state.ship.width * 0.85, state.ship.width / 2 + shot.width / 2);
    var laneScore = 1 - clamp(Math.abs((shot.x + shot.width / 2) - shipCenter) / laneWidth, 0, 1);
    var yProgress = clamp(shot.y / canvas.height, 0, 1);
    var distanceToShip = clamp(Math.max(0, state.ship.y - shot.y) / canvas.height, 0, 1);
    return laneScore * 3 + yProgress - distanceToShip * 0.35;
  }

  function dangerousPilotBullet(liveAliens) {
    var best = null;
    state.bullets.forEach(function (bullet) {
      liveAliens.forEach(function (alien) {
        if (bullet.y + bullet.height < alien.y - 8) return;
        var laneScore = shotLaneOverlap(bullet, alien);
        var verticalScore = 1 - clamp(Math.abs(bullet.y - (alien.y + alien.height)) / canvas.height, 0, 1);
        var score = laneScore * 3 + verticalScore;
        if (!best || score > best.score) {
          best = { score: score, bullet: bullet, alien: alien };
        }
      });
    });
    return best;
  }

  function shotLaneOverlap(shot, actor) {
    var laneWidth = Math.max(actor.width * 0.85, actor.width / 2 + shot.width / 2);
    var distance = Math.abs((shot.x + shot.width / 2) - (actor.x + actor.width / 2));
    return 1 - clamp(distance / laneWidth, 0, 1);
  }

  function fleetProgress() {
    var liveAliens = state.aliens.filter(function (alien) { return alien.alive; });
    if (!liveAliens.length) return 0;
    return clamp(Math.max.apply(null, liveAliens.map(function (alien) { return alien.y; })) / canvas.height, 0, 1);
  }

  function nearestThreat() {
    var liveAliens = state.aliens.filter(function (alien) { return alien.alive; });
    if (!liveAliens.length) return null;
    liveAliens.sort(function (a, b) {
      return Math.abs((a.x + a.width / 2) - (state.ship.x + state.ship.width / 2)) -
        Math.abs((b.x + b.width / 2) - (state.ship.x + state.ship.width / 2));
    });
    return liveAliens[0];
  }

  function updateShip(dt) {
    var direction = 0;
    var verticalDirection = 0;
    if (keys.ArrowLeft || keys.KeyA) direction -= 1;
    if (keys.ArrowRight || keys.KeyD) direction += 1;
    if (keys.ArrowUp || keys.KeyW) verticalDirection -= 1;
    if (keys.ArrowDown || keys.KeyS) verticalDirection += 1;
    if (pointerX !== null) {
      var desired = pointerX - state.ship.width / 2;
      state.ship.x += (desired - state.ship.x) * Math.min(1, dt * 12);
    } else {
      state.ship.x += direction * state.ship.speed * dt;
    }
    wrapShipHorizontal();
    state.ship.y += verticalDirection * state.ship.verticalSpeed * dt;
    state.ship.y = clamp(state.ship.y, canvas.height - 170, canvas.height - 56);
    if (keys.Space) fire();
  }

  function wrapShipHorizontal() {
    if (state.ship.x + state.ship.width < 0) {
      state.ship.x = canvas.width;
    } else if (state.ship.x > canvas.width) {
      state.ship.x = -state.ship.width;
    }
  }

  function updateBullets(dt) {
    state.bullets.forEach(function (bullet) { bullet.y -= bullet.speed * dt; });
    state.enemyShots.forEach(function (shot) { shot.y += shot.speed * dt; });
    state.bullets = state.bullets.filter(function (bullet) { return bullet.y > -bullet.height; });
    state.enemyShots = state.enemyShots.filter(function (shot) { return shot.y < canvas.height + shot.height; });
  }

  function updateAliens(dt) {
    var liveAliens = state.aliens.filter(function (alien) { return alien.alive; });
    if (!liveAliens.length) return;
    var formationAliens = liveAliens.filter(function (alien) { return !alien.free; });
    liveAliens.forEach(function (alien) {
      alien.wobble += dt * 5;
      alien.shotCooldown = Math.max(0, (alien.shotCooldown || 0) - dt);
      alien.downCooldown = Math.max(0, (alien.downCooldown || 0) - dt);
      if (alien.free) {
        updateFreeAlien(alien, dt);
      } else {
        alien.x += state.fleetDirection * state.fleetSpeed * dt;
        wrapAlienHorizontal(alien);
        if (alien.loop) {
          alien.x += Math.cos(alien.wobble * 2.1) * 85 * dt;
          alien.y += Math.sin(alien.wobble * 2.1) * 45 * dt;
          wrapAlienHorizontal(alien);
        }
      }
      if (alien.alive && intersects(alien, state.ship)) {
        alien.alive = false;
        loseLife();
      } else if (alien.y + alien.height >= canvas.height) {
        alien.alive = false;
      }
    });
  }

  function updateFreeAlien(alien, dt) {
    var shipCenter = state.ship.x + state.ship.width / 2;
    if (alien.scatter) {
      alien.x += alien.scatter * (140 + state.wave * 12) * dt;
      alien.y += (55 + state.wave * 8) * dt;
    } else {
      alien.x += clamp(shipCenter - (alien.x + alien.width / 2), -1, 1) * (180 + state.wave * 16) * dt;
      alien.y += (130 + state.wave * 14) * dt;
    }
    wrapAlienHorizontal(alien);
    if (alien.y > canvas.height - 130) {
      alien.free = false;
      alien.scatter = 0;
      alien.loop = false;
    }
  }

  function wrapAlienHorizontal(alien) {
    if (alien.x + alien.width < 0) {
      alien.x = canvas.width;
    } else if (alien.x > canvas.width) {
      alien.x = -alien.width;
    }
  }

  function maybeAlienFire(dt) {
    if (enemyModel) return;
    var liveAliens = state.aliens.filter(function (alien) { return alien.alive; });
    var shooters = liveAliens.filter(function (alien) {
      return alien.role === "butterfly" || alien.role === "boss";
    });
    if (!shooters.length) return;
    var chance = (0.18 + state.wave * 0.035) * dt;
    if (Math.random() < chance) {
      var alien = shooters[Math.floor(Math.random() * shooters.length)];
      fireAlienShot(alien);
    }
  }

  function fireAlienShot(alien) {
    if (!canAlienFire(alien)) return false;
    state.enemyShots.push({
      x: alien.x + alien.width / 2 - 3,
      y: alien.y + alien.height,
      width: 6,
      height: 16,
      // Spec-aligned (was 210 in the browser vs 230 in the trainer -- a silent
      // sim-to-real drift the single-source-of-truth seam exposed).
      speed: SPEC.enemyShot.speed + state.wave * SPEC.enemyShot.speedPerWave
    });
    alien.shotCooldown = SPEC.timing.enemyShipShotCooldown;
    return true;
  }

  function checkHits() {
    state.bullets.forEach(function (bullet) {
      state.aliens.forEach(function (alien) {
        if (!alien.alive || !intersects(bullet, alien)) return;
        alien.alive = false;
        bullet.y = -100;
        state.score += 50 * state.wave;
        updateHud();
      });
    });

    state.enemyShots.forEach(function (shot) {
      if (!intersects(shot, state.ship)) return;
      shot.y = canvas.height + 100;
      loseLife();
    });
  }

  function loseLife() {
    if (shake > 0.25) return;
    state.lives -= 1;
    shake = 0.45;
    state.enemyShots = [];
    state.ship.x = canvas.width / 2 - state.ship.width / 2;
    state.ship.y = canvas.height - 72;
    if (state.lives <= 0) {
      endGame("Game over");
    } else {
      updateHud();
    }
  }

  function fire() {
    if (fireCooldown > 0 || !running) return;
    state.bullets.push({
      x: state.ship.x + state.ship.width / 2 - 3,
      y: state.ship.y - 14,
      width: 6,
      height: 18,
      speed: SPEC.bullet.speed
    });
    fireCooldown = 0.17;
  }

  function draw(time) {
    ctx.save();
    if (shake > 0) {
      ctx.translate(Math.sin(time * 80) * 4, Math.cos(time * 70) * 3);
    }
    ctx.clearRect(-10, -10, canvas.width + 20, canvas.height + 20);
    drawBackground(time);
    drawShip();
    drawAliens(time);
    drawProjectiles();
    if (!running) drawOverlay(state.message || "Press Start");
    ctx.restore();
  }

  function drawBackground(time) {
    var gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, "#07091f");
    gradient.addColorStop(1, "#130626");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(52, 245, 255, 0.18)";
    for (var i = 0; i < 80; i += 1) {
      var x = (i * 127 + Math.floor(time * 18)) % canvas.width;
      var y = (i * 71 + Math.floor(time * 34)) % canvas.height;
      ctx.fillRect(x, y, i % 5 === 0 ? 2 : 1, i % 5 === 0 ? 2 : 1);
    }
  }

  function drawShip() {
    if (assets.ship.complete && assets.ship.naturalWidth) {
      ctx.drawImage(assets.ship, state.ship.x, state.ship.y, state.ship.width, state.ship.height);
      return;
    }
    ctx.fillStyle = "#34f5ff";
    ctx.beginPath();
    ctx.moveTo(state.ship.x + state.ship.width / 2, state.ship.y);
    ctx.lineTo(state.ship.x + state.ship.width, state.ship.y + state.ship.height);
    ctx.lineTo(state.ship.x, state.ship.y + state.ship.height);
    ctx.closePath();
    ctx.fill();
  }

  function drawAliens(time) {
    state.aliens.forEach(function (alien) {
      if (!alien.alive) return;
      var y = alien.y + Math.sin(time * 4 + alien.wobble) * 3;
      var image = alien.type ? assets.alienB : assets.alienA;
      if (image.complete && image.naturalWidth) {
        ctx.drawImage(image, alien.x, y, alien.width, alien.height);
        drawAlienRoleMark(alien, y);
        return;
      }
      ctx.fillStyle = alien.type ? "#ff4fc3" : "#83ff8f";
      ctx.fillRect(alien.x, y, alien.width, alien.height);
      drawAlienRoleMark(alien, y);
    });
  }

  function drawAlienRoleMark(alien, y) {
    var colors = {
      bee: "#ffd166",
      butterfly: "#ff4fc3",
      boss: "#34f5ff"
    };
    ctx.save();
    ctx.strokeStyle = colors[alien.role] || "#f5f7ff";
    ctx.lineWidth = alien.role === "boss" ? 3 : 2;
    ctx.beginPath();
    ctx.arc(alien.x + alien.width / 2, y + 4, alien.role === "boss" ? 9 : 6, Math.PI, 0);
    ctx.stroke();
    if (alien.free || alien.loop) {
      ctx.fillStyle = colors[alien.role] || "#f5f7ff";
      ctx.fillRect(alien.x + alien.width / 2 - 7, y - 5, 14, 3);
    }
    ctx.restore();
  }

  function drawProjectiles() {
    ctx.fillStyle = "#ffd166";
    state.bullets.forEach(function (bullet) {
      ctx.fillRect(bullet.x, bullet.y, bullet.width, bullet.height);
    });
    ctx.fillStyle = "#ff4fc3";
    state.enemyShots.forEach(function (shot) {
      ctx.fillRect(shot.x, shot.y, shot.width, shot.height);
    });
  }

  function drawOverlay(message) {
    ctx.fillStyle = "rgba(5, 7, 18, 0.72)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f5f7ff";
    ctx.textAlign = "center";
    ctx.font = "900 52px system-ui, sans-serif";
    ctx.fillText(message, canvas.width / 2, canvas.height / 2 - 18);
    ctx.font = "700 22px system-ui, sans-serif";
    ctx.fillStyle = "#aeb7da";
    ctx.fillText("Start the game or toggle the trained pilot", canvas.width / 2, canvas.height / 2 + 26);
  }

  function intersects(a, b) {
    return a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function canvasXFromEvent(event) {
    var rect = canvas.getBoundingClientRect();
    return (event.clientX - rect.left) * (canvas.width / rect.width);
  }

  document.addEventListener("keydown", function (event) {
    keys[event.code] = true;
    if (event.code === "Space") event.preventDefault();
    if (event.code === "KeyP") {
      pilot = !pilot;
      updateHud();
    }
  });

  document.addEventListener("keyup", function (event) {
    keys[event.code] = false;
  });

  canvas.addEventListener("pointerdown", function (event) {
    pointerX = canvasXFromEvent(event);
    fire();
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", function (event) {
    if (event.buttons) pointerX = canvasXFromEvent(event);
  });

  canvas.addEventListener("pointerup", function () {
    pointerX = null;
  });

  startButton.addEventListener("click", startGame);
  resetButton.addEventListener("click", function () {
    state = createInitialState();
    running = false;
    updateEnemyModelForMode();
    updateHud();
  });
  aiButton.addEventListener("click", function () {
    pilot = !pilot;
    updateHud();
  });
  if (enemyModeButton) {
    enemyModeButton.addEventListener("click", function () {
      enemyMode = enemyMode === "progression" ? "manual" : "progression";
      updateEnemyModelForMode();
      updateHud();
    });
  }
  if (versionNodes.pilotSlider) {
    versionNodes.pilotSlider.addEventListener("input", function (event) {
      selectVersion("pilot", event.target.value);
    });
  }
  if (versionNodes.enemySlider) {
    versionNodes.enemySlider.addEventListener("input", function (event) {
      selectVersion("enemies", event.target.value);
    });
  }

  requestAnimationFrame(function (now) {
    lastTime = now;
    requestAnimationFrame(loop);
  });
})();
