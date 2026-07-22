"""Conservative area-of-effect sap targeting."""

from __future__ import annotations

import math

from cortex_arena.config import AgentConfig, Position, chebyshev
from cortex_arena.models import SapOrder, UnitSnapshot
from cortex_arena.world import WorldModel


class CombatPlanner:
    """Assign sap shots only when there is observable tactical value.

    Opponents move simultaneously, so firing at every visible ship wastes large
    amounts of energy.  The policy focuses on stacks, low-energy targets, and
    ships contesting possible scoring tiles.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def plan(
        self,
        world: WorldModel,
        units: list[UnitSnapshot],
        *,
        points_behind: bool,
    ) -> dict[int, SapOrder]:
        enemies = [
            enemy for enemy in world.visible_opponents.values() if enemy.energy >= 0
        ]
        if not enemies:
            return {}

        target_positions = sorted({enemy.position for enemy in enemies})
        candidates: list[tuple[float, Position, int]] = []
        for target in target_positions:
            direct = [enemy for enemy in enemies if enemy.position == target]
            splash = [
                enemy
                for enemy in enemies
                if enemy.position != target and chebyshev(enemy.position, target) <= 1
            ]
            strategic = 0.0
            if world.score_known[target] == world.SCORING:
                strategic = 2.0
            elif world.candidate_mask[target]:
                strategic = 0.8
            utility = 1.4 * len(direct) + 0.5 * len(splash) + strategic
            weakest = min(enemy.energy for enemy in direct)
            reliable = (
                len(direct) >= 2
                or strategic > 0
                or weakest <= math.ceil(self.config.unit_sap_cost * 1.25)
                or points_behind
            )
            if reliable:
                candidates.append((utility, target, max(enemy.energy for enemy in direct)))

        orders: dict[int, SapOrder] = {}
        available = {
            unit.unit_id: unit
            for unit in units
            if unit.energy >= self.config.unit_sap_cost
        }
        for utility, target, target_energy in sorted(candidates, reverse=True):
            eligible = [
                unit
                for unit in available.values()
                if abs(unit.position[0] - target[0]) <= self.config.unit_sap_range
                and abs(unit.position[1] - target[1]) <= self.config.unit_sap_range
            ]
            eligible.sort(
                key=lambda unit: (
                    chebyshev(unit.position, target),
                    -unit.energy,
                    unit.unit_id,
                )
            )
            shots = min(
                len(eligible),
                max(1, target_energy // self.config.unit_sap_cost + 1),
                3,
            )
            for unit in eligible[:shots]:
                orders[unit.unit_id] = SapOrder(unit.unit_id, target, utility)
                available.pop(unit.unit_id, None)
        return orders
