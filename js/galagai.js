(function () {
  "use strict";

  var canvas = document.getElementById("game");
  var ctx = canvas.getContext("2d");
  var scoreNode = document.getElementById("score");
  var waveNode = document.getElementById("wave");
  var livesNode = document.getElementById("lives");
  var bestNode = document.getElementById("best");
  var startButton = document.getElementById("start-button");
  var aiButton = document.getElementById("ai-button");
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
  var pilotModel = null;
  var enemyModel = null;
  var pilotVersions = [];
  var enemyVersions = [];
  var activePilotVersion = 0;
  var activeEnemyVersion = 0;
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

  function loadPilotModel() {
    fetch(modelManifestUrl)
      .then(function (response) {
        if (!response.ok) throw new Error("model unavailable");
        return response.json();
      })
      .then(function (model) {
        if (!isUsableManifest(model)) return;
        pilotVersions = extractPilotVersions(model);
        enemyVersions = extractEnemyVersions(model);
        activePilotVersion = Math.max(0, pilotVersions.length - 1);
        activeEnemyVersion = Math.max(0, enemyVersions.length - 1);
        applySelectedVersions();
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

  function extractPilotVersions(model) {
    var versions = model.versions && Array.isArray(model.versions.pilot)
      ? model.versions.pilot.filter(isLoadableVersion)
      : [];
    if (versions.length) return versions;
    return [model];
  }

  function extractEnemyVersions(model) {
    var versions = model.versions && Array.isArray(model.versions.enemies)
      ? model.versions.enemies.filter(isLoadableVersion)
      : [];
    if (versions.length) return versions;
    return isUsableModel(model.enemies) ? [model.enemies] : [];
  }

  function applySelectedVersions() {
    configureVersionSlider(versionNodes.pilotSlider, pilotVersions, activePilotVersion);
    configureVersionSlider(versionNodes.enemySlider, enemyVersions, activeEnemyVersion);
    loadSelectedVersion("pilot");
    loadSelectedVersion("enemies");
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
    }
    applySelectedVersions();
    updateHud();
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
        x: canvas.width / 2 - 32,
        y: canvas.height - 72,
        width: 64,
        height: 48,
        speed: 470
      },
      bullets: [],
      enemyShots: [],
      aliens: createFleet(1),
      fleetDirection: 1,
      fleetDrop: 18,
      fleetSpeed: 38
    };
  }

  function createFleet(wave) {
    var aliens = [];
    var columns = Math.min(9, 6 + wave);
    var rows = Math.min(5, 3 + Math.floor(wave / 2));
    var gapX = 78;
    var gapY = 54;
    var startX = (canvas.width - (columns - 1) * gapX) / 2 - 24;
    for (var row = 0; row < rows; row += 1) {
      for (var col = 0; col < columns; col += 1) {
        aliens.push({
          x: startX + col * gapX,
          y: 74 + row * gapY,
          width: 48,
          height: 34,
          type: (row + col) % 2,
          alive: true,
          wobble: Math.random() * Math.PI * 2
        });
      }
    }
    return aliens;
  }

  function startGame() {
    state = createInitialState();
    running = true;
    lastTime = performance.now();
    state.message = "";
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
    updateCheckpointPanel();
  }

  function updateCheckpointPanel() {
    var pilotEntry = pilotVersions[activePilotVersion] || null;
    var enemyEntry = enemyVersions[activeEnemyVersion] || null;
    var metrics = modelMetrics(pilotModel || pilotEntry);
    var selfPlay = metrics && metrics.selfPlay ? metrics.selfPlay : null;
    var latest = selfPlay && selfPlay.latest ? selfPlay.latest : null;

    checkpointNodes.modelName.textContent = pilotEntry ? modelLabel(pilotModel || pilotEntry) : "heuristic";
    checkpointNodes.enemyModelName.textContent = enemyEntry ? modelLabel(enemyModel || enemyEntry) : "scripted";
    checkpointNodes.evalAccuracy.textContent = metrics ? percent(metrics.evalAccuracy) : "--";
    checkpointNodes.selfPlayRound.textContent = latest ? latest.round : "--";
    checkpointNodes.trainedSide.textContent = latest ? latest.trained : "--";
    checkpointNodes.enemyAction.textContent = enemyModel ? enemyAction : "--";
    checkpointNodes.pilotWinRate.textContent = latest ? percent(latest.enemyWinRate) : "--";
    checkpointNodes.enemyPressure.textContent = latest ? percent(latest.enemyDropRate) : "--";
    if (versionNodes.pilotLabel) {
      versionNodes.pilotLabel.textContent = pilotEntry ? versionLabel(pilotModel || pilotEntry, activePilotVersion, pilotVersions.length) : "none";
    }
    if (versionNodes.enemyLabel) {
      versionNodes.enemyLabel.textContent = enemyEntry ? versionLabel(enemyModel || enemyEntry, activeEnemyVersion, enemyVersions.length) : "none";
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
      state.fleetSpeed += 13;
      state.score += 250;
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
    var action = predictModelAction(pilotModel, "stay");
    keys.ArrowLeft = action === "left";
    keys.ArrowRight = action === "right";
    if (action === "fire") fire();
  }

  function predictModelAction(model, fallback) {
    if (!model) return fallback;
    var features = modelFeatures(model);
    var scores = model.network ? evaluateNetwork(model.network, features) : null;
    var bestAction = fallback;
    var bestScoreForAction = -Infinity;
    model.actions.forEach(function (action, actionIndex) {
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

  function modelFeatures(model) {
    var features = pilotFeatures();
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

    var action = predictModelAction(enemyModel, "drop");
    enemyAction = action;
    if (action === "drift_left") {
      state.fleetDirection = -1;
    } else if (action === "drift_right") {
      state.fleetDirection = 1;
    } else if (action === "drop" && enemyDropCooldown <= 0) {
      liveAliens.forEach(function (alien) { alien.y += state.fleetDrop * 0.65; });
      enemyDropCooldown = 1.08;
    } else if (action === "drop") {
      enemyAction = "drop-cooldown";
    } else if (action === "fire" && enemyShotCooldown <= 0) {
      fireAlienShot(nearestThreat() || liveAliens[Math.floor(Math.random() * liveAliens.length)]);
      enemyShotCooldown = 0.42;
    } else if (action === "fire") {
      enemyAction = "fire-cooldown";
    }
    enemyThinkCooldown = Math.max(0.12, 0.24 - state.wave * 0.008);
    updateHud();
  }

  function pilotFeatures() {
    var shipCenter = state.ship.x + state.ship.width / 2;
    var target = nearestThreat();
    var targetDx = 0;
    var alienCount = 0;
    if (target) {
      targetDx = clamp(((target.x + target.width / 2) - shipCenter) / (canvas.width / 2), -1, 1);
      alienCount = state.aliens.filter(function (alien) { return alien.alive; }).length / 45;
    }

    var closestShot = state.enemyShots.slice().sort(function (a, b) {
      return Math.abs((a.x + a.width / 2) - shipCenter) - Math.abs((b.x + b.width / 2) - shipCenter);
    })[0];
    var threatDx = closestShot
      ? clamp(((closestShot.x + closestShot.width / 2) - shipCenter) / (canvas.width / 2), -1, 1)
      : 1;
    var threatY = closestShot ? clamp(closestShot.y / canvas.height, 0, 1) : 0;

    return [
      targetDx,
      Math.abs(targetDx),
      threatDx,
      threatY,
      fireCooldown <= 0 ? 1 : 0,
      clamp(alienCount, 0, 1),
      clamp(state.wave / 10, 0, 1),
      enemyDropCooldown <= 0 ? 1 : 0,
      enemyShotCooldown <= 0 ? 1 : 0,
      fleetProgress(),
      clamp(state.lives / 3, 0, 1)
    ];
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
    if (keys.ArrowLeft || keys.KeyA) direction -= 1;
    if (keys.ArrowRight || keys.KeyD) direction += 1;
    if (pointerX !== null) {
      var desired = pointerX - state.ship.width / 2;
      state.ship.x += (desired - state.ship.x) * Math.min(1, dt * 12);
    } else {
      state.ship.x += direction * state.ship.speed * dt;
    }
    state.ship.x = clamp(state.ship.x, 18, canvas.width - state.ship.width - 18);
    if (keys.Space) fire();
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
    var hitEdge = liveAliens.some(function (alien) {
      var nextX = alien.x + state.fleetDirection * state.fleetSpeed * dt;
      return nextX < 16 || nextX + alien.width > canvas.width - 16;
    });
    if (hitEdge) {
      state.fleetDirection *= -1;
      liveAliens.forEach(function (alien) { alien.y += state.fleetDrop; });
    }
    liveAliens.forEach(function (alien) {
      alien.x += state.fleetDirection * state.fleetSpeed * dt;
      alien.wobble += dt * 5;
      if (alien.y + alien.height >= state.ship.y) {
        loseLife();
      }
    });
  }

  function maybeAlienFire(dt) {
    if (enemyModel) return;
    var liveAliens = state.aliens.filter(function (alien) { return alien.alive; });
    if (!liveAliens.length) return;
    var chance = (0.18 + state.wave * 0.035) * dt;
    if (Math.random() < chance) {
      var alien = liveAliens[Math.floor(Math.random() * liveAliens.length)];
      fireAlienShot(alien);
    }
  }

  function fireAlienShot(alien) {
    if (!alien) return;
    state.enemyShots.push({
      x: alien.x + alien.width / 2 - 3,
      y: alien.y + alien.height,
      width: 6,
      height: 16,
      speed: 210 + state.wave * 20
    });
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
      speed: 620
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
        return;
      }
      ctx.fillStyle = alien.type ? "#ff4fc3" : "#83ff8f";
      ctx.fillRect(alien.x, y, alien.width, alien.height);
    });
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
    updateHud();
  });
  aiButton.addEventListener("click", function () {
    pilot = !pilot;
    updateHud();
  });
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
