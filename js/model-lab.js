/*
 * Brain selector: choose which AI technique drives the Pilot and the Enemies,
 * with an inline explainer (design, implementation, pros, cons, caveats) for the
 * chosen technique.
 *
 * Technique descriptions come from window.GALAGAI_DESIGNS (defined in
 * js/architectures.js) so the selector and the gallery never disagree. Which
 * technique is actually *live* (has exported weights driving the game) is read
 * from window.GalagAIRuntime, published by js/galagai.js after the model
 * manifest loads. Techniques without an artifact stay visible in the gallery,
 * but the live selector only enables exported brains.
 */
(function () {
  "use strict";

  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  ready(function () {
    var selectorRoot = document.getElementById("brain-selectors");
    var explainerRoot = document.getElementById("brain-explainer");
    var featureRoot = document.getElementById("feature-selectors");
    var featureExplainerRoot = document.getElementById("feature-explainer");
    var designs = window.GALAGAI_DESIGNS || [];
    if (!selectorRoot || !explainerRoot || !designs.length) return;

    var byId = {};
    designs.forEach(function (d) {
      byId[d.id] = d;
    });

    var FEATURE_REPRESENTATIONS = [
      {
        id: "grid-v1",
        name: "Map grid + scalars",
        status: "current",
        accent: "#34f5ff",
        target: "Spatial map view",
        summary: "Current exported models see an 8-channel map of the arena plus a few scalar flags. This preserves bullet lanes, wrap-around edges, and fleet layout.",
        notes: [
          "Best current default for collision-heavy play.",
          "More inputs than a vector, but it keeps spatial information explicit.",
          "Matches the checked-in DQN, QR-DQN, PPO, and MaskablePPO artifacts."
        ]
      },
      {
        id: "compact-v1",
        name: "Computed vector",
        status: "planned",
        accent: "#83ff8f",
        target: "Small engineered feature vector",
        summary: "A compact vector would feed only chosen facts such as nearest threats, cooldowns, lane danger, role counts, and fleet pressure.",
        notes: [
          "Fastest representation to train and export.",
          "Good for ablations against the map encoder.",
          "Can miss spatial patterns when many actors overlap."
        ]
      },
      {
        id: "hybrid-v1",
        name: "Hybrid map + vector",
        status: "planned",
        accent: "#ffd166",
        target: "Map view plus computed tactical scalars",
        summary: "A hybrid encoder keeps the map for spatial context and adds a compact tactical vector for high-signal facts the MLP should not have to rediscover.",
        notes: [
          "Likely best next upgrade after a compact-vector baseline.",
          "Needs trainer/export metadata so JS knows both input blocks.",
          "Costs more than a vector but less guesswork than map-only learning."
        ]
      },
      {
        id: "entity-set-v1",
        name: "Entity set / attention",
        status: "planned",
        accent: "#ff4fc3",
        target: "Variable-size per-entity rows",
        summary: "An entity encoder would process each alien, bullet, and shot as rows, then pool or attend over the set instead of flattening the map.",
        notes: [
          "Best fit for variable fleet size and many simultaneous threats.",
          "Deep-Set mean pooling is still hand-rollable in JS.",
          "Attention variants probably need ONNX Runtime Web."
        ]
      }
    ];

    var FEATURE_BUDGETS = [
      {
        id: "current",
        name: "Current artifact budget",
        status: "current",
        detail: "10,758 inputs from grid-v1",
        notes: ["No retrain needed; this is what the exported weights expect."]
      },
      {
        id: "32",
        name: "Tiny vector",
        status: "planned",
        detail: "32 engineered values",
        notes: ["Good smoke test for feature selection and overfitting checks."]
      },
      {
        id: "64",
        name: "Small vector",
        status: "planned",
        detail: "64 engineered values",
        notes: ["Best first compact baseline: enough room for threats, lanes, counts, cooldowns, and role signals."]
      },
      {
        id: "128",
        name: "Wide vector",
        status: "planned",
        detail: "128 engineered values",
        notes: ["Useful if 64 loses too much spatial detail but a full grid is still too heavy."]
      }
    ];

    var featureSelection = { representation: "grid-v1", budget: "current" };

    // Which technique is live per side. Defaults to dqn (the only exported
    // family today); galagai.js refines this from the manifest's algorithm.
    function liveId(side) {
      // Prefer the brain galagai.js is actually running for this side (per-side
      // mixing), then the manifest runtime, then a sensible default.
      var ga = window.GalagAI;
      if (ga && ga.currentBrain) {
        var cur = ga.currentBrain(side);
        if (cur && byId[cur]) return cur;
      }
      var rt = window.GalagAIRuntime || {};
      var id = side === "pilot" ? rt.pilotTechniqueId : rt.enemyTechniqueId;
      return byId[id] ? id : (byId.dqn ? "dqn" : designs[0].id);
    }

    function brainAvailable(id) {
      var ga = window.GalagAI;
      return Boolean(ga && ga.availableBrains && ga.availableBrains().indexOf(id) >= 0);
    }

    var selection = { pilot: liveId("pilot"), enemy: liveId("enemy") };

    function el(tag, className, text) {
      var node = document.createElement(tag);
      if (className) node.className = className;
      if (text != null) node.textContent = text;
      return node;
    }

    function statusWord(id, side) {
      if (id === liveId(side)) return "live";
      if (brainAvailable(id)) return "available";
      return byId[id] && byId[id].status === "planned" ? "planned" : "offline";
    }

    function optionLabel(id, side) {
      var design = byId[id] || {};
      var label = design.name || id;
      if (id === liveId(side)) return label + "  \u25cf live";
      if (brainAvailable(id)) return label + "  \u25cf exported";
      return label + (design.status === "planned" ? "  (planned)" : "  (not exported yet)");
    }

    function refreshSelectOptions(side) {
      var sel = document.getElementById("brain-" + side);
      if (!sel) return;
      Array.prototype.forEach.call(sel.options, function (opt) {
        var live = opt.value === liveId(side);
        opt.textContent = optionLabel(opt.value, side);
        opt.disabled = !live && !brainAvailable(opt.value);
      });
    }

    function buildSelect(side, labelText) {
      var wrap = el("label", "brain-select");
      wrap.appendChild(el("span", null, labelText));
      var select = el("select");
      select.id = "brain-" + side;
      designs.forEach(function (d) {
        var opt = el("option", null, optionLabel(d.id, side));
        opt.value = d.id;
        opt.disabled = d.id !== liveId(side) && !brainAvailable(d.id);
        select.appendChild(opt);
      });
      select.value = selection[side];
      select.addEventListener("change", function () {
        var tech = select.value;
        selection[side] = tech;
        var ga = window.GalagAI;
        if (ga && ga.setBrain && brainAvailable(tech)) {
          // Swap the live brain for this side, then refresh the explainer.
          ga.setBrain(side, tech)
            .then(function () { renderExplainer(side, tech); })
            .catch(function () { renderExplainer(side, tech); });
        } else {
          renderExplainer(side, tech);
        }
      });
      wrap.appendChild(select);
      return wrap;
    }

    function findById(items, id) {
      for (var i = 0; i < items.length; i += 1) {
        if (items[i].id === id) return items[i];
      }
      return items[0];
    }

    function featureOptionLabel(item) {
      return item.name + (item.status === "current" ? "  \u25cf current" : "  (planned)");
    }

    function buildFeatureSelect(id, labelText, items, value) {
      var wrap = el("label", "brain-select feature-select");
      wrap.appendChild(el("span", null, labelText));
      var select = el("select");
      select.id = id;
      items.forEach(function (item) {
        var opt = el("option", null, featureOptionLabel(item));
        opt.value = item.id;
        select.appendChild(opt);
      });
      select.value = value;
      select.addEventListener("change", function () {
        if (id === "feature-representation") {
          featureSelection.representation = select.value;
          if (select.value === "grid-v1") featureSelection.budget = "current";
          if (select.value !== "grid-v1" && featureSelection.budget === "current") featureSelection.budget = "64";
          var budgetSelect = document.getElementById("feature-budget");
          if (budgetSelect) budgetSelect.value = featureSelection.budget;
        } else {
          featureSelection.budget = select.value;
          if (select.value === "current") featureSelection.representation = "grid-v1";
          var repSelect = document.getElementById("feature-representation");
          if (repSelect) repSelect.value = featureSelection.representation;
        }
        renderFeatureExplainer();
      });
      wrap.appendChild(select);
      return wrap;
    }

    function product(values) {
      if (!Array.isArray(values) || !values.length) return null;
      return values.reduce(function (total, value) { return total * Number(value || 0); }, 1);
    }

    function inputCount(shape, scalars) {
      var grid = product(shape);
      if (grid == null && scalars == null) return null;
      return (grid || 0) + (Number(scalars) || 0);
    }

    function formatNumber(value) {
      if (value == null || !isFinite(value)) return "--";
      return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }

    function formatObservation(info) {
      if (!info || !info.featureEncoding) return "unknown";
      var shape = Array.isArray(info.frameShape) ? info.frameShape.join("x") : "vector";
      var scalars = info.scalarFeatureCount != null ? " + " + info.scalarFeatureCount + " scalars" : "";
      var total = inputCount(info.frameShape, info.scalarFeatureCount);
      return info.featureEncoding + " · " + shape + scalars + " · " + formatNumber(total) + " inputs";
    }

    function liveFeatureInfo(side) {
      var rt = window.GalagAIRuntime || {};
      var prefix = side === "pilot" ? "pilot" : "enemy";
      return {
        featureEncoding: rt[prefix + "FeatureEncoding"] || "grid-v1",
        frameShape: rt[prefix + "FrameShape"] || [8, 28, 48],
        scalarFeatureCount: rt[prefix + "ScalarFeatureCount"] != null ? rt[prefix + "ScalarFeatureCount"] : 6
      };
    }

    function addFact(dl, label, value) {
      var row = el("div", "feature-fact");
      row.appendChild(el("dt", null, label));
      row.appendChild(el("dd", null, value));
      dl.appendChild(row);
    }

    function renderFeatureExplainer() {
      if (!featureExplainerRoot) return;
      var rep = findById(FEATURE_REPRESENTATIONS, featureSelection.representation);
      var budget = findById(FEATURE_BUDGETS, featureSelection.budget);
      var planned = rep.status !== "current" || budget.status !== "current";
      featureExplainerRoot.innerHTML = "";
      featureExplainerRoot.style.setProperty("--feature-accent", rep.accent || "#34f5ff");

      var head = el("div", "feature-explainer-head");
      head.appendChild(el("p", "brain-eyebrow", "Observation design"));
      head.appendChild(el("h3", "brain-name", rep.name));
      head.appendChild(el("p", "brain-tagline", rep.summary));
      featureExplainerRoot.appendChild(head);

      var notice = el("p", "feature-notice " + (planned ? "feature-notice-planned" : "feature-notice-current"));
      notice.textContent = planned
        ? "Planning spec: train and publish a matching artifact before this representation can drive play."
        : "Current export: checked-in weights expect this observation shape.";
      featureExplainerRoot.appendChild(notice);

      var facts = el("dl", "feature-facts");
      addFact(facts, "Pilot live input", formatObservation(liveFeatureInfo("pilot")));
      addFact(facts, "Enemy live input", formatObservation(liveFeatureInfo("enemy")));
      addFact(facts, "Selected target", rep.target);
      addFact(facts, "Feature budget", budget.detail);
      featureExplainerRoot.appendChild(facts);

      var notes = el("ul", "feature-notes");
      rep.notes.concat(budget.notes).forEach(function (note) {
        notes.appendChild(el("li", null, note));
      });
      featureExplainerRoot.appendChild(notes);
    }

    function list(title, items, cssClass) {
      if (!items || !items.length) return null;
      var box = el("div", "brain-list " + cssClass);
      box.appendChild(el("h4", null, title));
      var ul = el("ul");
      items.forEach(function (item) {
        ul.appendChild(el("li", null, item));
      });
      box.appendChild(ul);
      return box;
    }

    function renderExplainer(side, id) {
      var d = byId[id];
      explainerRoot.innerHTML = "";
      if (!d) return;
      explainerRoot.style.setProperty("--brain-accent", d.accent || "#34f5ff");

      var head = el("div", "brain-explainer-head");
      head.appendChild(el("p", "brain-eyebrow", (side === "pilot" ? "Pilot brain" : "Enemy brain") + " · " + d.family));
      head.appendChild(el("h3", "brain-name", d.name));
      head.appendChild(el("p", "brain-tagline", d.tagline || ""));
      explainerRoot.appendChild(head);

      var word = statusWord(id, side);
      var notice = el("p", "brain-notice brain-notice-" + word);
      if (id === liveId(side)) {
        notice.textContent = "Live: exported weights are driving the " + side + " right now.";
      } else if (brainAvailable(id)) {
        notice.textContent = "Exported — select it to load this technique for the " + side + ".";
      } else {
        var live = byId[liveId(side)];
        notice.textContent =
          "Not exported yet — the live " + (live ? live.name : "model") +
          " keeps driving the " + side + ". Train this technique to enable it.";
      }
      explainerRoot.appendChild(notice);

      // Design + Implementation summaries straight from the gallery record.
      (d.sections || []).forEach(function (section) {
        if (!section.p || !section.p.length) return;
        explainerRoot.appendChild(el("h4", "brain-section-title", section.h));
        explainerRoot.appendChild(el("p", "brain-section-body", section.p[0]));
      });

      var grid = el("div", "brain-tradeoffs");
      [list("Pros", d.pros, "brain-pros"), list("Cons", d.cons, "brain-cons")].forEach(function (n) {
        if (n) grid.appendChild(n);
      });
      explainerRoot.appendChild(grid);
      var caveats = list("Caveats", d.caveats, "brain-caveats");
      if (caveats) explainerRoot.appendChild(caveats);

      var more = el("a", "brain-more", "Read the full design →");
      more.href = "#/design/" + d.id;
      explainerRoot.appendChild(more);
    }

    var grid = el("div", "brain-select-grid");
    grid.appendChild(buildSelect("pilot", "Pilot technique"));
    grid.appendChild(buildSelect("enemy", "Enemy technique"));
    selectorRoot.appendChild(grid);

    if (featureRoot) {
      var featureGrid = el("div", "feature-select-grid");
      featureGrid.appendChild(buildFeatureSelect(
        "feature-representation",
        "Observation type",
        FEATURE_REPRESENTATIONS,
        featureSelection.representation
      ));
      featureGrid.appendChild(buildFeatureSelect(
        "feature-budget",
        "Feature budget",
        FEATURE_BUDGETS,
        featureSelection.budget
      ));
      featureRoot.appendChild(featureGrid);
      renderFeatureExplainer();
    }

    // Default view: explain the pilot's live technique.
    renderExplainer("pilot", selection.pilot);

    // Re-mark live state if galagai.js publishes runtime info after us.
    window.addEventListener("galagai:runtime", function () {
      ["pilot", "enemy"].forEach(function (side) {
        refreshSelectOptions(side);
      });
      renderFeatureExplainer();
    });
  });
})();
