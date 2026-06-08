# `alien_invasion/` is the legacy pygame implementation

**Status: deprecated, kept for reference. Not the game you play or train.**

This directory is the original local pygame version of the game. The live product
is now two pieces that share one set of rules:

- **Browser runtime** — `js/galagai.js` (+ `js/encoder.js`), what visitors play.
- **Headless trainer** — `tools/train_static_pilot.py`, which produces the agents.

Both read the canonical rules from **`game_spec.json`** and the canonical
observation layout from the shared encoder, pinned together by
`tests/test_game_spec_contract.py` and `tests/test_encoder_*.py`.

This pygame code is a **third implementation** of the same game and has diverged
from those two:

- different canvas (1360×768 vs the canonical 960×560),
- the old 4-action DQN action space (trained models now use 6 pilot / 8 enemy
  actions), and
- `game_functions.get_state()` screen-scrapes the rendered display instead of
  reading game state directly.

Architecture review recommendation **D** was: apply the deletion test — either
retire this directory, or make it a fourth *adapter* of `game_spec.json` rather
than a hand-maintained fork. Until that decision is made, treat it as read-only
reference. The recursive `Alien.update()` bug (which could exceed Python's
recursion limit) has been fixed in place so the reference at least runs cleanly.

Do not copy constants out of here — they are stale. Use `game_spec.json`.
