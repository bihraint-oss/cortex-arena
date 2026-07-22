"""Hierarchical role allocation for exploration, scoring, and interception."""

from __future__ import annotations

from collections.abc import Iterable

from cortex_arena.config import AgentConfig, Position, manhattan
from cortex_arena.models import Assignment, Role, UnitSnapshot
from cortex_arena.world import WorldModel


class TacticalPlanner:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def assign(
        self,
        world: WorldModel,
        units: list[UnitSnapshot],
        *,
        points_behind: bool,
    ) -> dict[int, Assignment]:
        """Allocate every visible friendly ship to one explainable role."""

        assignments: dict[int, Assignment] = {}
        available = {unit.unit_id: unit for unit in units}
        reserved_targets: set[Position] = set()

        # A unit already earning on a confirmed tile should not be pulled away.
        confirmed = [
            tile for tile in world.confirmed_scoring_tiles() if world.is_passable(tile)
        ]
        confirmed_set = set(confirmed)
        likely = [
            tile
            for tile in world.policy_scoring_tiles()
            if world.is_passable(tile) and tile not in confirmed_set
        ]
        likely.sort()
        likely_set = set(likely)
        for unit in units:
            if unit.position in confirmed_set and unit.position not in reserved_targets:
                assignments[unit.unit_id] = Assignment(
                    unit.unit_id,
                    Role.HARVESTER,
                    unit.position,
                    "hold confirmed scoring tile",
                )
                reserved_targets.add(unit.position)
                available.pop(unit.unit_id, None)
            elif unit.position in likely_set and unit.position not in reserved_targets:
                assignments[unit.unit_id] = Assignment(
                    unit.unit_id,
                    Role.HARVESTER,
                    unit.position,
                    "hold high-confidence scoring hypothesis",
                )
                reserved_targets.add(unit.position)
                available.pop(unit.unit_id, None)

        # Low-energy ships route toward a known positive field before they become
        # stranded. Holding a scoring tile remains more valuable than recharging.
        low_threshold = self.config.unit_move_cost * self.config.low_energy_moves
        for unit in sorted(available.values(), key=lambda item: (item.energy, item.unit_id)):
            if unit.energy >= low_threshold:
                continue
            target = world.best_energy_tile(
                unit.position, available_energy=unit.energy
            )
            if target is None:
                continue
            assignments[unit.unit_id] = Assignment(
                unit.unit_id, Role.RECHARGE, target, "energy reserve below safety margin"
            )
            reserved_targets.add(target)
            available.pop(unit.unit_id, None)

        # Fill unoccupied confirmed scoring tiles with nearest available ships.
        open_scoring = [tile for tile in confirmed if tile not in reserved_targets]
        self._greedy_allocate(
            available,
            open_scoring,
            assignments,
            reserved_targets,
            role=Role.HARVESTER,
            reason="occupy confirmed scoring tile",
        )

        # Belief is intentionally weaker than proof, but exploiting a strong
        # hypothesis earns points while later zero-delta evidence can still
        # revoke it.
        open_likely = [tile for tile in likely if tile not in reserved_targets]
        self._greedy_allocate(
            available,
            open_likely,
            assignments,
            reserved_targets,
            role=Role.HARVESTER,
            reason="exploit high-confidence scoring hypothesis",
        )

        # Visible enemies near a scoring region receive an interceptor. Sapping is
        # decided separately; this role closes distance when a shot is unavailable.
        threats = [
            track.position
            for track in world.visible_opponents.values()
            if track.energy >= 0 and world.candidate_mask[track.position]
        ]
        if threats and available:
            self._greedy_allocate(
                available,
                threats[: max(1, len(available) // 3)],
                assignments,
                reserved_targets,
                role=Role.INTERCEPTOR,
                reason="contest opponent near relic",
            )

        # Test unknown relic-adjacent tiles. Distinct positions make point deltas
        # identifiable, allowing the world model to learn the hidden reward mask.
        unknown = [
            tile
            for tile in world.unknown_scoring_tiles()
            if world.is_passable(tile) and tile not in reserved_targets
        ]
        unknown.sort(
            key=lambda tile: (
                -float(world.score_belief[tile]),
                int(world.visits[tile]),
                tile,
            )
        )
        if available and unknown:
            # Keep the score equation closed: while unknown tiles remain around
            # an observed relic, route every unassigned ship into that footprint.
            # Outside scouts could themselves score via an unseen relic, making
            # the point delta impossible to attribute without false certainty.
            quota = len(available)
            self._greedy_allocate(
                available,
                unknown[:quota],
                assignments,
                reserved_targets,
                role=Role.PROSPECTOR,
                reason="identify hidden relic scoring tile",
            )

        # Remaining ships continually refresh fog-of-war. This is important in
        # early matches because new relic pairs can appear in previously seen areas.
        for unit in sorted(available.values(), key=lambda item: item.unit_id):
            target = world.frontier_target(unit.position, reserved_targets)
            assignments[unit.unit_id] = Assignment(
                unit.unit_id, Role.SCOUT, target, "maximize fresh sensor coverage"
            )
            reserved_targets.add(target)

        return assignments

    @staticmethod
    def _greedy_allocate(
        available: dict[int, UnitSnapshot],
        targets: Iterable[Position],
        assignments: dict[int, Assignment],
        reserved_targets: set[Position],
        *,
        role: Role,
        reason: str,
    ) -> None:
        for target in targets:
            if not available or target in reserved_targets:
                continue
            unit = min(
                available.values(),
                key=lambda item: (manhattan(item.position, target), -item.energy, item.unit_id),
            )
            assignments[unit.unit_id] = Assignment(unit.unit_id, role, target, reason)
            reserved_targets.add(target)
            available.pop(unit.unit_id, None)
