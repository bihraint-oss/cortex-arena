"""Risk-aware A* routing over a partially observed, drifting map."""

from __future__ import annotations

import heapq
import itertools

from cortex_arena.config import (
    DIRECTION_DELTAS,
    ActionType,
    AgentConfig,
    Position,
    TileType,
    manhattan,
)
from cortex_arena.world import WorldModel


class Pathfinder:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def next_action(
        self, world: WorldModel, start: Position, goal: Position
    ) -> tuple[ActionType, Position]:
        """Return the first safe step on an A* route to ``goal``."""

        if start == goal:
            return ActionType.WAIT, start

        frontier: list[tuple[float, int, Position]] = []
        sequence = itertools.count()
        heapq.heappush(frontier, (float(manhattan(start, goal)), next(sequence), start))
        came_from: dict[Position, tuple[Position, ActionType]] = {}
        cost_so_far: dict[Position, float] = {start: 0.0}

        found = False
        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                found = True
                break
            for action, neighbor in self._neighbors(world, current):
                new_cost = cost_so_far[current] + self._step_cost(world, neighbor)
                if new_cost >= cost_so_far.get(neighbor, float("inf")):
                    continue
                cost_so_far[neighbor] = new_cost
                priority = new_cost + manhattan(neighbor, goal)
                heapq.heappush(frontier, (priority, next(sequence), neighbor))
                came_from[neighbor] = current, action

        if not found:
            return self._greedy_fallback(world, start, goal)

        current = goal
        while came_from[current][0] != start:
            current = came_from[current][0]
        _, action = came_from[current]
        return action, current

    def _neighbors(
        self, world: WorldModel, position: Position
    ) -> list[tuple[ActionType, Position]]:
        x, y = position
        result: list[tuple[ActionType, Position]] = []
        for action, (dx, dy) in DIRECTION_DELTAS.items():
            target = x + dx, y + dy
            if world.is_passable(target):
                result.append((action, target))
        return result

    def _step_cost(self, world: WorldModel, position: Position) -> float:
        tile = world.current_tile_type(position)
        cost = 1.0
        if tile < 0:
            cost += self.config.unknown_tile_cost
        elif tile == TileType.NEBULA:
            cost += self.config.nebula_cost
        energy = world.known_energy(position)
        if energy < 0:
            cost += -energy * self.config.negative_energy_cost
        # A recently observed opponent makes a tile tactically risky, but not
        # impossible: a high-energy collision can still be intentional.
        for opponent in world.visible_opponents.values():
            if opponent.energy < 0:
                continue
            if opponent.position == position:
                cost += 4.0
            elif manhattan(opponent.position, position) == 1:
                # The exact void factor is hidden. Scale risk monotonically with
                # the visible opposing energy instead of pretending it is known.
                cost += opponent.energy * 0.02
        return cost

    def _greedy_fallback(
        self, world: WorldModel, start: Position, goal: Position
    ) -> tuple[ActionType, Position]:
        choices = self._neighbors(world, start)
        if not choices:
            return ActionType.WAIT, start
        return min(
            choices,
            key=lambda pair: manhattan(pair[1], goal) + self._step_cost(world, pair[1]),
        )


def direction_to(start: Position, target: Position) -> ActionType:
    """Convert adjacent coordinates into the official discrete move action."""

    delta = target[0] - start[0], target[1] - start[1]
    for action, expected in DIRECTION_DELTAS.items():
        if delta == expected:
            return action
    return ActionType.WAIT
