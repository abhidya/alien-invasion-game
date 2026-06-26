/*
 * GalagAI architecture gallery.
 *
 * Renders one "design + implementation" card per training method/architecture
 * researched for this project, plus a hash-routed detail view (#/design/<id>)
 * so every option has its own shareable, page-like URL. Pure static DOM, no
 * dependencies -- runs as-is on GitHub Pages.
 *
 * Content is grounded in the deep-research briefs; each card cites primary
 * sources. The `status` field reflects the current repo state:
 *   live      - default exported artifact loaded on first page load
 *   exported  - exported artifact is available in the brain selector
 *   trainable - implemented in the trainer, but no artifact is checked in yet
 *   planned   - designed, not yet wired into the trainer/export path
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
            "Input: an 8x28x48 map grid plus 6 scalar flags/counts (10,758 inputs, featureEncoding=grid-v1).",
            "Network: MLP [64, 64] with ReLU; output = one Q-value per action.",
            "Replay buffer + target network stabilize the bootstrapped TD target."
          ]
        },
        {
          h: "Implementation",
          p: [
            "Trained headlessly in the same arcade loop the browser runs, against frozen snapshots of the opposing agent. The learned Q-network is a stack of dense layers, so export is trivial: serialize weights/biases to JSON and evaluate a matmul+ReLU forward pass in ~30 lines of JavaScript.",
            "This is exactly what powers the playable demo above today (schema v17, wrap-aware grid features)."
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
      status: "planned",
      accent: "#83ff8f",
      tagline: "Rainbow's load-bearing pieces, minus the export pain. All training-only — the deployed net is unchanged.",
      facts: [
        { label: "Library", value: "Planned custom SB3 DQN variant" },
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
            "Planned as a custom DQN subclass: a few lines in the target computation (n-step + Double), a prioritized buffer, and an optional dueling head. Because none of this has to change the acting network's shape, the existing JSON export and hand-rolled JS forward pass can keep working.",
            "This is the lowest-risk upgrade path once wired: better sample efficiency and stability with no deploy-time runtime change."
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
      status: "exported",
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
      status: "exported",
      accent: "#ff4fc3",
      tagline: "The strongest general self-play default. Learns a stochastic policy that handles a moving opponent better than DQN's greedy value chase.",
      facts: [
        { label: "Library", value: "stable-baselines3 (core)" },
        { label: "Action space", value: "Discrete" },
        { label: "Browser runtime", value: "Hand-rolled JS actor" },
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
      id: "a2c",
      name: "Advantage Actor-Critic",
      family: "Policy-gradient / actor-critic",
      status: "trainable",
      accent: "#ff4fc3",
      tagline: "PPO's simpler synchronous cousin. Same stochastic actor and identical plain-MLP export, lighter to run.",
      facts: [
        { label: "Library", value: "stable-baselines3 (core)" },
        { label: "Action space", value: "Discrete" },
        { label: "Browser runtime", value: "Hand-rolled JS" },
        { label: "Export", value: "Actor MLP → logits" }
      ],
      sections: [
        {
          h: "Design",
          p: [
            "A2C is the synchronous A3C-style actor-critic: it estimates an advantage and nudges a stochastic policy toward better-than-average actions. It shares PPO's on-policy footing — no replay buffer, so no stale experience from obsolete self-play opponents and no replay-pickle disk cost.",
            "It omits PPO's clipped trust region, so it is simpler and faster per step but a touch less stable; a reasonable lightweight fallback when PPO's robustness is not required."
          ],
          ul: [
            "Identical export to PPO: the actor is a plain state→logits MLP.",
            "On-policy and reproducible; pairs well with the unified self-play loop."
          ]
        },
        {
          h: "Implementation",
          p: [
            "Train with SB3 A2C + MlpPolicy. Export only the actor (mlp_extractor.policy_net + action_net) and argmax in JS — the same path PPO uses, so the browser runtime is unchanged."
          ]
        }
      ],
      sources: [
        { label: "A2C / A3C overview", url: "https://apxml.com/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic/a2c-a3c" },
        { label: "DQN vs PPO vs A2C study", url: "https://arxiv.org/html/2407.14151v1" }
      ]
    },
    {
      id: "maskable-ppo",
      name: "MaskablePPO",
      family: "Policy-gradient (action-masked)",
      status: "exported",
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
            "The manifest records that action masking is required; the JS runtime rebuilds the same legal-action mask from the browser game rules before choosing an action."
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
      status: "planned",
      accent: "#83ff8f",
      tagline: "Gradient-free. Evolve a population of policies by match outcome — and export the winning weights straight to the browser.",
      facts: [
        { label: "Library", value: "Planned ES / NEAT path" },
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
            "Would run ES/NEAT over the headless arcade env, evaluating each candidate against a pool of frozen opponents. The champion's weights could export to the same JSON the hand-rolled JS evaluator already consumes — so an evolved fixed-MLP policy would plug into the demo with minimal runtime change.",
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

  // Pros / cons / caveats per design, merged into the records above. Kept as a
  // separate map so the trade-off content sits in one readable block and is
  // reused verbatim by both the gallery detail view and the brain selector
  // (js/model-lab.js).
  var TRADEOFFS = {
    dqn: {
      pros: [
        "Live today: real exported weights play in the browser.",
        "Trivial export — dense weights to JSON, ~30 lines of JS inference.",
        "Sample-efficient via replay buffer; zero runtime dependencies."
      ],
      cons: [
        "Deterministic argmax policy is brittle against a shifting self-play opponent.",
        "Q-overestimation can destabilise bootstrapped self-play.",
        "No notion of action legality — can waste capacity on illegal moves."
      ],
      caveats: [
        "The current schema uses wrap-aware grid features; older schema v14 artifacts used legacy linear features.",
        "The browser must reproduce the trainer's observation exactly (see js/encoder.js)."
      ]
    },
    "dqn-plus": {
      pros: [
        "n-step + PER lift sample efficiency with no deploy-time change.",
        "Double/Dueling cut Q-overestimation cheaply.",
        "Acting network shape unchanged — existing JSON export just works."
      ],
      cons: [
        "Extra training-time machinery (prioritised buffer, n-step targets).",
        "Still a deterministic value policy at deploy.",
        "PER adds hyperparameters to tune."
      ],
      caveats: [
        "All gains are training-only; the deployed net is identical to DQN.",
        "Planned trainer/export work; no artifact exists yet."
      ]
    },
    "qr-dqn": {
      pros: [
        "Models the full return distribution — richer, more robust signal.",
        "Beats categorical C51 on Atari while still exporting as an MLP.",
        "Browser inference is mean-over-quantiles then argmax (a few lines)."
      ],
      cons: [
        "Output layer is |A|×N quantiles — wider net, more memory.",
        "Still value-based and deterministic at deploy.",
        "Requires sb3-contrib rather than core SB3."
      ],
      caveats: [
        "The manifest must record the quantile count so JS folds the output right.",
        "Exported artifact exists under js/brains/qr-dqn; retrain when you want it refreshed to the latest schema and opponent pool."
      ]
    },
    ppo: {
      pros: [
        "Stochastic policy handles a moving opponent better than greedy DQN.",
        "On-policy — never reuses stale experience from obsolete opponents.",
        "Strong, hyperparameter-robust self-play default."
      ],
      cons: [
        "Less sample-efficient than replay-based methods.",
        "Must export only the actor and replay SB3 preprocessing in JS.",
        "Critic is wasted compute at deploy time."
      ],
      caveats: [
        "Observation preprocessing is NOT bundled in the export — mirror it in JS.",
        "Exported artifact exists under js/brains/ppo; the browser evaluates actor logits directly in JS."
      ]
    },
    a2c: {
      pros: [
        "Simpler and faster per step than PPO; reproducible under fixed seeds.",
        "On-policy — no replay buffer, so no stale experience and no replay-pickle disk cost.",
        "Identical plain-MLP actor export to PPO; browser runtime unchanged."
      ],
      cons: [
        "No clipped trust region, so updates are a touch less stable than PPO.",
        "Less sample-efficient than replay-based value methods.",
        "Critic is wasted compute at deploy time."
      ],
      caveats: [
        "Mirror SB3 observation preprocessing in JS, same as PPO.",
        "No A2C artifact published yet."
      ]
    },
    "maskable-ppo": {
      pros: [
        "Knows which actions are legal — directly fits this game's role asymmetry.",
        "Masking is a valid policy gradient, not a reward hack.",
        "Actor stays a plain MLP, so static export is unaffected."
      ],
      cons: [
        "Requires an action_masks() method from the env.",
        "Browser must rebuild the exact mask before argmax.",
        "sb3-contrib dependency."
      ],
      caveats: [
        "Mask logic must match between trainer and browser or behaviour diverges.",
        "Exported artifact exists under js/brains/maskable-ppo; manifest actionMasking tells JS to rebuild masks from game rules."
      ]
    },
    "neuro-es": {
      pros: [
        "Gradient-free — sidesteps reward-shaping headaches entirely.",
        "Fitness is match outcome; parallelises trivially across workers.",
        "Champion exports as plain weight arrays — cleanest possible browser path."
      ],
      cons: [
        "Sample-inefficient relative to gradient methods.",
        "Population evaluation is compute-heavy.",
        "NEAT topology search adds its own complexity."
      ],
      caveats: [
        "Use a historical-opponent archive to avoid cycling, same as gradient self-play.",
        "Planned only; no ES/NEAT trainer or artifact is wired in yet."
      ]
    },
    "deepset-attn": {
      pros: [
        "Permutation-invariant over a variable-size fleet — the 'correct' representation.",
        "Mean-pool Deep-Set stays hand-rollable in JS.",
        "Attention adds alien–alien reasoning when needed."
      ],
      cons: [
        "Needs a per-entity observation refactor in the env first.",
        "Attention variants require ONNX Runtime Web (WASM), not plain JS.",
        "Only pays off if the fixed nearest-N hack is shown to hurt."
      ],
      caveats: [
        "Marked planned — not wired into the pipeline.",
        "Attention export is heavier; load it lazily so MLP-only visitors pay nothing."
      ]
    }
  };

  DESIGNS.forEach(function (design) {
    var extra = TRADEOFFS[design.id];
    if (extra) {
      design.pros = extra.pros;
      design.cons = extra.cons;
      design.caveats = extra.caveats;
    }
  });

  // Shared with js/model-lab.js (the pilot/enemy brain selector) so the gallery
  // and the selector describe each technique from one source.
  window.GALAGAI_DESIGNS = DESIGNS;

  var STATUS_LABELS = {
    live: "Live in demo",
    exported: "Exported",
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

    function tradeoffList(title, items, cssClass) {
      if (!items || !items.length) return;
      panel.appendChild(el("h3", "arch-section-title", title));
      var list = el("ul", "arch-section-list " + cssClass);
      items.forEach(function (item) {
        list.appendChild(el("li", null, item));
      });
      panel.appendChild(list);
    }
    tradeoffList("Pros", design.pros, "arch-pros");
    tradeoffList("Cons", design.cons, "arch-cons");
    tradeoffList("Caveats", design.caveats, "arch-caveats");

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
