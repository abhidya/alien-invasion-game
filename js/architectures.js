/*
 * GalagAI architecture gallery.
 *
 * Renders one "design + implementation" card per training method/architecture
 * researched for this project, plus a hash-routed detail view (#/design/<id>)
 * so every option has its own shareable, page-like URL. Pure static DOM, no
 * dependencies -- runs as-is on GitHub Pages.
 *
 * Content is grounded in the deep-research briefs; each card cites primary
 * sources. The `status` field reflects what is wired into the training +
 * checkpoint pipeline today:
 *   live      - currently exported and playable in the demo above
 *   trainable - implemented in the unified pipeline; retrain to publish
 *   planned   - designed, not yet wired into the trainer
 */
(function () {
  "use strict";

  var DESIGNS = [
    {
      id: "dqn",
      name: "Deep Q-Network",
      family: "Value-based",
      status: "live",
      accent: "#34f5ff",
      tagline: "The baseline. A dense Q-network scores every action; the agent plays the argmax.",
      facts: [
        { label: "Library", value: "stable-baselines3 (core)" },
        { label: "Action space", value: "Discrete" },
        { label: "Browser runtime", value: "Hand-rolled JS MLP" },
        { label: "Export", value: "Dense weights → JSON" }
      ],
      sections: [
        {
          h: "Design",
          p: [
            "DQN learns a value function Q(state, action) that estimates the total future reward of taking each action. The policy is greedy: run the state through the network and pick the action with the highest score (argmax).",
            "It is off-policy and sample-efficient thanks to a replay buffer that recycles past transitions, but it learns a deterministic policy via epsilon-greedy exploration, which can be brittle under the shifting opponents of self-play."
          ],
          ul: [
            "Input: the 27-dim engineered feature vector (relative threat positions, cooldowns, counts).",
            "Network: MLP [64, 64] with ReLU; output = one Q-value per action.",
            "Replay buffer + target network stabilize the bootstrapped TD target."
          ]
        },
        {
          h: "Implementation",
          p: [
            "Trained headlessly in the same arcade loop the browser runs, against frozen snapshots of the opposing agent. The learned Q-network is a stack of dense layers, so export is trivial: serialize weights/biases to JSON and evaluate a matmul+ReLU forward pass in ~30 lines of JavaScript.",
            "This is exactly what powers the playable demo above today (schema v14, legacy linear features; v15+ uses the wrap-aware features)."
          ],
          ul: [
            "Browser inference: zero dependencies, zero cold-start, iOS/Safari-safe.",
            "Deployment step: argmax over the output logits in JS."
          ]
        }
      ],
      sources: [
        { label: "SB3 DQN docs", url: "https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html" },
        { label: "SB3 model export (ONNX)", url: "https://stable-baselines3.readthedocs.io/en/master/guide/export.html" }
      ]
    },
    {
      id: "dqn-plus",
      name: "DQN + n-step / PER / Double",
      family: "Value-based (improved)",
      status: "trainable",
      accent: "#83ff8f",
      tagline: "Rainbow's load-bearing pieces, minus the export pain. All training-only — the deployed net is unchanged.",
      facts: [
        { label: "Library", value: "Custom SB3 DQN variant" },
        { label: "Action space", value: "Discrete" },
        { label: "Browser runtime", value: "Hand-rolled JS MLP" },
        { label: "Export", value: "Identical to DQN" }
      ],
      sections: [
        {
          h: "Design",
          p: [
            "Rainbow's ablation showed that of its six ingredients, multi-step returns and prioritized replay carry most of the gains, while Double and Dueling are cheap correctness tweaks. Crucially, all of these live in the loss and replay buffer — the acting network stays a plain state→Q[a] MLP."
          ],
          ul: [
            "n-step returns: back up reward over n steps for faster, lower-bias credit assignment — valuable for short ~360-step episodes.",
            "Prioritized Experience Replay (PER): sample high-TD-error transitions more often (median Atari 48%→106% over vanilla DQN).",
            "Double DQN: decouple action-selection from evaluation to cut Q-overestimation (a real risk in bootstrapped self-play).",
            "Dueling: split into value + advantage streams that recombine to Q[a] in-graph."
          ]
        },
        {
          h: "Implementation",
          p: [
            "Implemented as a custom DQN subclass: a few lines in the target computation (n-step + Double), a prioritized buffer, and an optional dueling head. Because none of this changes the acting network's shape, the existing JSON export and hand-rolled JS forward pass keep working with no changes.",
            "This is the lowest-risk upgrade path: better sample efficiency and stability for free at deploy time."
          ]
        }
      ],
      sources: [
        { label: "Rainbow (Hessel et al.)", url: "https://arxiv.org/abs/1710.02298" },
        { label: "Prioritized Experience Replay", url: "https://arxiv.org/pdf/1511.05952" },
        { label: "Dueling architectures", url: "https://arxiv.org/pdf/1511.06581" }
      ]
    },
    {
      id: "qr-dqn",
      name: "QR-DQN (Distributional)",
      family: "Value-based (distributional)",
      status: "trainable",
      accent: "#ffd166",
      tagline: "Learns the full distribution of returns, not just the mean. Best ready-made distributional method that still exports as an MLP.",
      facts: [
        { label: "Library", value: "sb3-contrib (MlpPolicy)" },
        { label: "Action space", value: "Discrete" },
        { label: "Browser runtime", value: "Hand-rolled JS (mean+argmax)" },
        { label: "Export", value: "MLP → quantiles" }
      ],
      sections: [
        {
          h: "Design",
          p: [
            "Instead of a single Q-value per action, QR-DQN predicts N quantiles of the return distribution per action via quantile regression. Modeling the spread of outcomes — not just the average — yields a richer, more robust learning signal and beats categorical C51 on Atari (~178% vs lower median human-normalized score).",
            "It is the only distributional method that is both available off-the-shelf and clean to export."
          ]
        },
        {
          h: "Implementation",
          p: [
            "Swap the trainer to sb3-contrib's QRDQN with MlpPolicy. The output layer is |A|×N quantiles; to act, reduce-mean over the N quantiles for each action, then argmax. Both reductions are a couple of extra lines in the JS forward pass (or an ONNX ReduceMean+ArgMax), so it still runs dependency-free in the browser.",
            "The manifest records the quantile count so the JS evaluator knows how to fold the output."
          ]
        }
      ],
      sources: [
        { label: "QR-DQN (Dabney et al.)", url: "https://arxiv.org/pdf/1710.10044" },
        { label: "sb3-contrib QR-DQN", url: "https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html" }
      ]
    },
    {
      id: "ppo",
      name: "Proximal Policy Optimization",
      family: "Policy-gradient / actor-critic",
      status: "trainable",
      accent: "#ff4fc3",
      tagline: "The strongest general self-play default. Learns a stochastic policy that handles a moving opponent better than DQN's greedy value chase.",
      facts: [
        { label: "Library", value: "stable-baselines3 (core)" },
        { label: "Action space", value: "Discrete" },
        { label: "Browser runtime", value: "Hand-rolled JS / ORT-Web" },
        { label: "Export", value: "Actor MLP → logits" }
      ],
      sections: [
        {
          h: "Design",
          p: [
            "PPO is an on-policy actor-critic method. The actor outputs a probability distribution over actions; a clipped surrogate objective keeps each update inside a soft trust region, giving the stability PPO is famous for and strong robustness to hyperparameters.",
            "Because it is on-policy, it never reuses stale experience generated against now-obsolete opponents — a structural advantage under the non-stationarity of self-play, where a value-method's replay buffer can hold misleading data."
          ],
          ul: [
            "Stochastic policy: better suited to competitive games than a deterministic argmax.",
            "Only the actor network is needed at deploy time; the critic is training-only."
          ]
        },
        {
          h: "Implementation",
          p: [
            "Train with SB3 PPO + MlpPolicy. Export only the actor subgraph (mlp_extractor + action_net) — a plain feedforward state→logits MLP — and take the argmax (or sample) in JS. A working static onnxruntime-web demo of an exported SB3 policy exists as a reference pattern.",
            "Remember to replicate SB3's observation preprocessing in JS; the export does not bundle it."
          ]
        }
      ],
      sources: [
        { label: "DQN vs PPO vs A2C study", url: "https://arxiv.org/html/2407.14151v1" },
        { label: "SB3 export → onnxruntime-web", url: "https://stable-baselines3.readthedocs.io/en/master/guide/export.html" }
      ]
    },
    {
      id: "maskable-ppo",
      name: "MaskablePPO",
      family: "Policy-gradient (action-masked)",
      status: "trainable",
      accent: "#34f5ff",
      tagline: "PPO that knows which actions are legal. The best fit for this game — only butterfly/boss roles can fire.",
      facts: [
        { label: "Library", value: "sb3-contrib" },
        { label: "Action space", value: "Discrete + mask" },
        { label: "Browser runtime", value: "Hand-rolled JS (mask logits)" },
        { label: "Export", value: "Actor MLP → masked logits" }
      ],
      sections: [
        {
          h: "Design",
          p: [
            "Many actions are invalid in specific states: in this game only certain enemy roles can fire, the drop has a cooldown, and edges constrain movement. Invalid-action masking sets the logits of illegal actions to −∞ before the softmax. This is a valid policy gradient (a state-dependent renormalization), not a hack, and it empirically improves both learning speed and final performance while scaling gracefully as the invalid set grows — unlike negative-reward penalties.",
            "This directly targets the structural-asymmetry failure mode we suspect in the current self-play loop."
          ]
        },
        {
          h: "Implementation",
          p: [
            "Train with sb3-contrib MaskablePPO, supplying an action_masks() method from the env (per-role fire legality, drop cooldown, etc.). The actor stays a plain MLP — masking happens outside the network — so static export is unaffected. At browser inference, reproduce the same mask in JS (set illegal logits to −1e9) before the argmax.",
            "The manifest carries a per-state mask spec so the JS runtime can rebuild it deterministically."
          ]
        }
      ],
      sources: [
        { label: "Invalid Action Masking", url: "https://arxiv.org/pdf/2006.14171" },
        { label: "sb3-contrib MaskablePPO", url: "https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html" }
      ]
    },
    {
      id: "neuro-es",
      name: "Evolution Strategies / NEAT",
      family: "Neuroevolution",
      status: "trainable",
      accent: "#83ff8f",
      tagline: "Gradient-free. Evolve a population of policies by match outcome — and export the winning weights straight to the browser.",
      facts: [
        { label: "Library", value: "ES / NEAT (custom · neataptic)" },
        { label: "Action space", value: "Discrete" },
        { label: "Browser runtime", value: "Hand-rolled JS / neataptic" },
        { label: "Export", value: "Plain weight arrays" }
      ],
      sections: [
        {
          h: "Design",
          p: [
            "Evolution Strategies optimize the policy weights directly by perturbing them with Gaussian noise and keeping what wins — no backprop, no gradients. Competitive coevolution is natural here: fitness is simply whether you beat the opponent. NEAT goes further and evolves the network topology alongside the weights.",
            "Because the objective is the match outcome, evolutionary methods sidestep reward-shaping headaches and parallelize trivially across workers."
          ],
          ul: [
            "ES yields a fixed-architecture MLP → the cleanest possible browser export (just weight arrays).",
            "NEAT genomes serialize to JSON and can even train/run fully client-side via neataptic.",
            "Apply the same historical-opponent archive as gradient self-play to avoid cycling."
          ]
        },
        {
          h: "Implementation",
          p: [
            "Run ES/NEAT over the headless arcade env, evaluating each candidate against a pool of frozen opponents. The champion's weights export to the same JSON the hand-rolled JS evaluator already consumes — so an evolved policy plugs into the demo with zero runtime changes.",
            "Best reached for if gradient self-play stalls, or when you want population diversity and trivial export in one package."
          ]
        }
      ],
      sources: [
        { label: "Evolution Strategies (Salimans et al.)", url: "https://www.emergentmind.com/papers/1703.03864" },
        { label: "neataptic (browser NEAT)", url: "https://github.com/wagenaartje/neataptic" }
      ]
    },
    {
      id: "deepset-attn",
      name: "Deep-Sets / Attention Encoder",
      family: "Architecture (set encoder)",
      status: "planned",
      accent: "#ffd166",
      tagline: "The 'correct' representation for a fleet: permutation-invariant over a variable number of aliens, instead of a fixed nearest-N hack.",
      facts: [
        { label: "Library", value: "Custom torch policy" },
        { label: "Action space", value: "Discrete" },
        { label: "Browser runtime", value: "JS (mean-pool) / ORT-Web (attention)" },
        { label: "Export", value: "Encoder + head → ONNX" }
      ],
      sections: [
        {
          h: "Design",
          p: [
            "The current features summarize the fleet with 'nearest-N + counts', which discards information when many aliens cluster. A Deep-Sets encoder embeds each alien with a shared MLP φ, pools the set (e.g. mean/sum) into a permutation-invariant summary, then decides with a head ρ — exactly the AlphaStar entity-encoder pattern, scaled down. Self-attention adds pairwise alien–alien reasoning on top.",
            "This is the one place where leaving the plain feature MLP genuinely pays off — but only if the fixed-N hack is shown to hurt."
          ]
        },
        {
          h: "Implementation",
          p: [
            "Per-entity tensors (one row per alien) feed φ; mean-pooling keeps a Deep-Set hand-rollable in JS (MLP + a mean). Attention variants need a softmax/matmul block, so those export to ONNX and run via onnxruntime-web (WASM, single-thread) — loaded lazily, only for this demo, so MLP-only visitors pay nothing.",
            "Marked planned: it requires a per-entity observation refactor in the env before it can be wired into the pipeline."
          ]
        }
      ],
      sources: [
        { label: "Deep Sets (Zaheer et al.)", url: "https://ar5iv.labs.arxiv.org/html/1703.06114" },
        { label: "ONNX Runtime Web deploy", url: "https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/tutorials/web/deploy.md" }
      ]
    }
  ];

  var STATUS_LABELS = {
    live: "Live in demo",
    trainable: "Trainable now",
    planned: "Planned"
  };

  var galleryRoot = document.getElementById("architecture-gallery");
  var detailRoot = document.getElementById("architecture-detail");
  if (!galleryRoot || !detailRoot) {
    return;
  }

  var designsById = {};
  DESIGNS.forEach(function (design) {
    designsById[design.id] = design;
  });

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text != null) {
      node.textContent = text;
    }
    return node;
  }

  function buildCard(design) {
    var card = el("button", "arch-card");
    card.type = "button";
    card.style.setProperty("--accent", design.accent);
    card.setAttribute("aria-label", "Open design and implementation for " + design.name);
    card.addEventListener("click", function () {
      location.hash = "#/design/" + design.id;
    });

    var top = el("div", "arch-card-top");
    top.appendChild(el("span", "arch-family", design.family));
    top.appendChild(el("span", "arch-status arch-status-" + design.status, STATUS_LABELS[design.status]));
    card.appendChild(top);

    card.appendChild(el("h3", "arch-name", design.name));
    card.appendChild(el("p", "arch-tagline", design.tagline));

    var facts = el("dl", "arch-facts");
    design.facts.forEach(function (fact) {
      var row = el("div", "arch-fact");
      row.appendChild(el("dt", null, fact.label));
      row.appendChild(el("dd", null, fact.value));
      facts.appendChild(row);
    });
    card.appendChild(facts);

    var cta = el("span", "arch-cta", "Design & implementation →");
    card.appendChild(cta);
    return card;
  }

  function renderGallery() {
    var grid = el("div", "arch-grid");
    DESIGNS.forEach(function (design) {
      grid.appendChild(buildCard(design));
    });
    galleryRoot.innerHTML = "";
    galleryRoot.appendChild(grid);
  }

  function buildDetail(design) {
    var panel = el("div", "arch-detail-panel");
    panel.style.setProperty("--accent", design.accent);
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    panel.setAttribute("aria-label", design.name + " design and implementation");
    panel.tabIndex = -1;

    var header = el("div", "arch-detail-header");
    var heading = el("div", "arch-detail-heading");
    heading.appendChild(el("span", "arch-family", design.family));
    heading.appendChild(el("h2", "arch-detail-title", design.name));
    heading.appendChild(el("span", "arch-status arch-status-" + design.status, STATUS_LABELS[design.status]));
    header.appendChild(heading);

    var close = el("button", "arch-close", "Close ✕");
    close.type = "button";
    close.addEventListener("click", closeDetail);
    header.appendChild(close);
    panel.appendChild(header);

    panel.appendChild(el("p", "arch-detail-tagline", design.tagline));

    var facts = el("dl", "arch-detail-facts");
    design.facts.forEach(function (fact) {
      var row = el("div", "arch-fact");
      row.appendChild(el("dt", null, fact.label));
      row.appendChild(el("dd", null, fact.value));
      facts.appendChild(row);
    });
    panel.appendChild(facts);

    design.sections.forEach(function (section) {
      panel.appendChild(el("h3", "arch-section-title", section.h));
      (section.p || []).forEach(function (paragraph) {
        panel.appendChild(el("p", "arch-section-text", paragraph));
      });
      if (section.ul && section.ul.length) {
        var list = el("ul", "arch-section-list");
        section.ul.forEach(function (item) {
          list.appendChild(el("li", null, item));
        });
        panel.appendChild(list);
      }
    });

    if (design.sources && design.sources.length) {
      panel.appendChild(el("h3", "arch-section-title", "Sources"));
      var sourceList = el("ul", "arch-source-list");
      design.sources.forEach(function (source) {
        var li = el("li");
        var link = el("a", null, source.label);
        link.href = source.url;
        link.target = "_blank";
        link.rel = "noopener";
        li.appendChild(link);
        sourceList.appendChild(li);
      });
      panel.appendChild(sourceList);
    }

    return panel;
  }

  function openDetail(design) {
    detailRoot.innerHTML = "";
    var backdrop = el("div", "arch-detail-backdrop");
    backdrop.addEventListener("click", function (event) {
      if (event.target === backdrop) {
        closeDetail();
      }
    });
    var panel = buildDetail(design);
    backdrop.appendChild(panel);
    detailRoot.appendChild(backdrop);
    detailRoot.hidden = false;
    document.body.classList.add("arch-detail-open");
    panel.focus();
  }

  function closeDetail() {
    if (detailRoot.hidden) {
      return;
    }
    detailRoot.hidden = true;
    detailRoot.innerHTML = "";
    document.body.classList.remove("arch-detail-open");
    if (/^#\/design\//.test(location.hash)) {
      // Drop the detail route without scrolling.
      history.replaceState(null, "", location.pathname + location.search + "#architectures");
    }
  }

  function syncFromHash() {
    var match = /^#\/design\/([\w-]+)$/.exec(location.hash);
    if (match && designsById[match[1]]) {
      openDetail(designsById[match[1]]);
    } else {
      closeDetail();
    }
  }

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      closeDetail();
    }
  });
  window.addEventListener("hashchange", syncFromHash);

  renderGallery();
  syncFromHash();
})();
