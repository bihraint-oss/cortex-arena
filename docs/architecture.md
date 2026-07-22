# Architecture

Cortex Arena uses a bounded perception–planning–action loop. It never asks a language model to emit executable Python, and it never calls a hidden game-state API beyond the observation officially exposed to each competitor.

## Decision loop

1. `WorldModel.update` merges the current team observation into persistent memory.
2. Newly observed map cells and objectives are mirrored using the environment's anti-diagonal symmetry.
3. The change in team points is first compared with distinct occupied tiles inside known relic footprints. Definitive evidence marks tiles scoring or non-scoring; unexplained positive points become evidence for an unseen relic.
4. `TacticalPlanner.assign` gives each visible unit one role and one target.
5. `CombatPlanner.plan` may override movement with a bounded sap order when the target is tactically reliable.
6. `Pathfinder.next_action` finds one risk-priced A* step toward the assigned target.
7. `Agent.act` validates offsets and emits the fixed `(max_units, 3)` integer matrix expected by Lux.

## State ownership

| Component | Owns | Does not own |
|---|---|---|
| `WorldModel` | observations, beliefs, visit counts, opponent tracks | policy decisions |
| `TacticalPlanner` | role and target selection | map mutation or movement syntax |
| `Pathfinder` | one-step safe route decisions | long-term strategy |
| `CombatPlanner` | sap target and shot allocation | movement |
| `Agent` | orchestration, validation, telemetry | environment internals |

Keeping those boundaries makes synthetic unit tests useful: a pathfinding regression does not require JAX compilation or a 505-step episode.

## Drift-safe memory

The official agent protocol does not expose terrain-drift direction, drift cadence, energy-node motion, nebula drain, or the exact opponent void factor. Cortex Arena therefore does not reconstruct those values from privileged environment state. Relics and objective evidence persist, while terrain is used for routing only in the observation that revealed it. Older coordinates remain useful as exploration history but degrade to unknown instead of becoming permanent obstacles.

Energy-node movement has an additional one-turn observation lag in the official environment. A visible change against the immediately preceding observation identifies the post-drift refresh; only those changed, currently visible values are eligible as recharge targets. An unchanged field is treated as uncertain. Nebula is rejected for low-energy recharge routes because its exact drain is hidden.

## Hidden scoring-tile inference

Each relic owns an unknown 5×5 binary scoring mask. Suppose the team score increases by `d` and friendly ships occupy `n` distinct, still-unknown tiles inside observed relic footprints:

- if `d = 0`, all `n` tiles are proven non-scoring;
- if `d = n`, all `n` tiles are proven scoring;
- otherwise each tile receives soft evidence `d / n` until later observations disambiguate it.

Already confirmed scoring tiles are subtracted before reasoning about the remaining unknowns. Duplicated ships on one tile count once, matching the environment's scoring rule. Every conclusion is mirrored because the official generator mirrors relic configurations.

Unrelated scouts are excluded from this equation so that normal exploration does not dilute the evidence. If the known footprints cannot explain a positive score delta, the remaining occupied positions receive candidate evidence for a relic that has spawned but is still hidden by fog. Negative conclusions are periodically reopened during possible relic-spawn windows, preventing an old mask from permanently hiding a later overlap.

The model keeps mathematical certainty (`score_known`) separate from policy confidence (`score_belief`). When known-footprint and outside positions can both explain a point, no tile is labelled proven. A known-footprint explanation may still receive a strong prior and be exploited as a reversible hypothesis; a later zero-point observation removes it. This distinction avoids turning a useful strategy preference into a false factual claim.

## Planning priorities

The current deterministic priority order is:

1. hold confirmed scoring tiles;
2. recharge ships below a configurable move reserve;
3. fill unoccupied confirmed scoring tiles;
4. intercept visible enemies contesting relic regions;
5. probe distinct unknown candidate tiles;
6. maximize fresh or stale sensor coverage.

This is deliberately inspectable. Set `CORTEX_DEBUG=1` to print one compact `CORTEX_TRACE` JSON record per decision without changing stdout, which is reserved for the runner protocol.

## Extension seam

The agent currently consumes the Lux observation dictionary directly. A future environment adapter should normalize another permitted game into:

- friendly units with stable IDs, positions, and resources;
- visible opponent tracks;
- a traversability and resource grid;
- objectives and team progress;
- a small validated action vocabulary.

The 0 A.D. R28 HTTP reinforcement-learning interface is the planned full-RTS adapter. A screen-and-input adapter is intentionally not part of the default architecture.
