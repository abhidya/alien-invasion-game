/*
 * Brain selector: choose which AI technique drives the Pilot and the Enemies,
 * with an inline explainer (design, implementation, pros, cons, caveats) for the
 * chosen technique.
 *
 * Technique descriptions come from window.GALAGAI_DESIGNS (defined in
 * js/architectures.js) so the selector and the gallery never disagree. Which
 * technique is actually *live* (has exported weights driving the game) is read
 * from window.GalagAIRuntime, published by js/galagai.js after the model
 * manifest loads. Techniques without an artifact are selectable for inspection
 * but show a clear "not exported yet" notice -- the live model keeps playing.
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
    var designs = window.GALAGAI_DESIGNS || [];
    if (!selectorRoot || !explainerRoot || !designs.length) return;

    var byId = {};
    designs.forEach(function (d) {
      byId[d.id] = d;
    });

    // Which technique is live per side. Defaults to dqn (the only exported
    // family today); galagai.js refines this from the manifest's algorithm.
    function liveId(side) {
      var rt = window.GalagAIRuntime || {};
      var id = side === "pilot" ? rt.pilotTechniqueId : rt.enemyTechniqueId;
      return byId[id] ? id : (byId.dqn ? "dqn" : designs[0].id);
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
      return byId[id] && byId[id].status === "planned" ? "planned" : "offline";
    }

    function buildSelect(side, labelText) {
      var wrap = el("label", "brain-select");
      wrap.appendChild(el("span", null, labelText));
      var select = el("select");
      select.id = "brain-" + side;
      designs.forEach(function (d) {
        var opt = el("option", null, d.name + (d.id === liveId(side) ? "  ● live" : ""));
        opt.value = d.id;
        select.appendChild(opt);
      });
      select.value = selection[side];
      select.addEventListener("change", function () {
        selection[side] = select.value;
        renderExplainer(side, select.value);
      });
      wrap.appendChild(select);
      return wrap;
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
      if (word === "live") {
        notice.textContent = "Live: exported weights are driving the " + side + " right now.";
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

    // Default view: explain the pilot's live technique.
    renderExplainer("pilot", selection.pilot);

    // Re-mark live state if galagai.js publishes runtime info after us.
    window.addEventListener("galagai:runtime", function () {
      ["pilot", "enemy"].forEach(function (side) {
        var sel = document.getElementById("brain-" + side);
        if (!sel) return;
        Array.prototype.forEach.call(sel.options, function (opt) {
          var base = (byId[opt.value] || {}).name || opt.value;
          opt.textContent = base + (opt.value === liveId(side) ? "  ● live" : "");
        });
      });
    });
  });
})();
