"""Registry of trainable RL algorithms for the GalagAI self-play pipeline.

This module is intentionally dependency-free: it imports neither torch nor
stable-baselines3, so the publisher, the manifest writer, and the tests can
reason about algorithm metadata without loading the heavy training stack. The
trainer (tools/train_static_pilot.py) maps each spec to a concrete
stable-baselines3 / sb3-contrib class at run time.

Every algorithm here deploys to client-side inference: its acting network is a
plain MLP whose forward pass the browser already runs by hand. The differences
that matter for export are captured in the spec:

* ``export_modules`` - the policy sub-module path(s), in order, whose Linear
  layers make up the exportable state -> output MLP. DQN exposes ``q_net``;
  QR-DQN ``quantile_net``; the actor-critic methods concatenate the shared
  ``mlp_extractor.policy_net`` with the ``action_net`` head.
* ``output_head`` - how the browser folds the network output into a chosen
  action: ``q-values``/``logits`` take a straight argmax; ``quantiles`` reduces
  the |A| x N quantile output by mean over quantiles, then argmax.
* ``action_masking`` - whether illegal actions are masked before the argmax
  (the mask is applied outside the network, in the env at train time and in JS
  at deploy time, so it never changes the exported weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    display_name: str
    family: str
    sb3_module: str
    sb3_class: str
    policy_name: str
    export_modules: tuple[str, ...]
    output_head: str
    browser_runtime: str
    manifest_algorithm: str
    technique: str
    action_masking: bool = False
    off_policy: bool = True
    extra_kwargs: dict[str, object] = field(default_factory=dict)


DEFAULT_ALGORITHM = "dqn"

ALGORITHMS: dict[str, AlgorithmSpec] = {
    "dqn": AlgorithmSpec(
        key="dqn",
        display_name="Deep Q-Network",
        family="value-based",
        sb3_module="stable_baselines3",
        sb3_class="DQN",
        policy_name="MlpPolicy",
        export_modules=("q_net",),
        output_head="q-values",
        browser_runtime="js-mlp",
        manifest_algorithm="stable-baselines3-dqn",
        technique="dqn",
        off_policy=True,
    ),
    "qrdqn": AlgorithmSpec(
        key="qrdqn",
        display_name="QR-DQN (distributional)",
        family="value-based",
        sb3_module="sb3_contrib",
        sb3_class="QRDQN",
        policy_name="MlpPolicy",
        export_modules=("quantile_net",),
        output_head="quantiles",
        browser_runtime="js-mlp",
        manifest_algorithm="sb3-contrib-qrdqn",
        technique="qr-dqn",
        off_policy=True,
    ),
    "ppo": AlgorithmSpec(
        key="ppo",
        display_name="Proximal Policy Optimization",
        family="policy-gradient",
        sb3_module="stable_baselines3",
        sb3_class="PPO",
        policy_name="MlpPolicy",
        export_modules=("mlp_extractor.policy_net", "action_net"),
        output_head="logits",
        browser_runtime="js-mlp",
        manifest_algorithm="stable-baselines3-ppo",
        technique="ppo",
        off_policy=False,
    ),
    "a2c": AlgorithmSpec(
        key="a2c",
        display_name="Advantage Actor-Critic",
        family="policy-gradient",
        sb3_module="stable_baselines3",
        sb3_class="A2C",
        policy_name="MlpPolicy",
        export_modules=("mlp_extractor.policy_net", "action_net"),
        output_head="logits",
        browser_runtime="js-mlp",
        manifest_algorithm="stable-baselines3-a2c",
        technique="a2c",
        off_policy=False,
    ),
    "maskable-ppo": AlgorithmSpec(
        key="maskable-ppo",
        display_name="MaskablePPO",
        family="policy-gradient",
        sb3_module="sb3_contrib",
        sb3_class="MaskablePPO",
        policy_name="MlpPolicy",
        export_modules=("mlp_extractor.policy_net", "action_net"),
        output_head="logits",
        browser_runtime="js-mlp",
        manifest_algorithm="sb3-contrib-maskable-ppo",
        technique="maskable-ppo",
        action_masking=True,
        off_policy=False,
    ),
}


def algorithm_keys() -> list[str]:
    """Return the selectable algorithm keys in registration order."""
    return list(ALGORITHMS)


def get_algorithm(key: str | None) -> AlgorithmSpec:
    """Look up an algorithm spec, defaulting to DQN when key is None."""
    resolved = key or DEFAULT_ALGORITHM
    try:
        return ALGORITHMS[resolved]
    except KeyError as error:
        raise ValueError(
            f"Unknown algorithm {resolved!r}. Choose one of: {', '.join(algorithm_keys())}."
        ) from error


def checkpoint_dir_name(algorithm: str, schema_version: int) -> str:
    """Suggested checkpoint directory for an algorithm + schema version.

    DQN keeps the historical ``galagai-balanced-v<schema>`` name so existing
    runs and tests are unaffected; other algorithms get an algorithm-suffixed
    sibling so their checkpoints never collide on the same path.
    """
    base = f"galagai-balanced-v{schema_version}"
    if algorithm == DEFAULT_ALGORITHM:
        return base
    return f"{base}-{algorithm}"
