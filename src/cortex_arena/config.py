"""Typed configuration and game constants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

Position = tuple[int, int]


class TileType(IntEnum):
    """Tile identifiers used by the Lux AI Season 3 environment."""

    EMPTY = 0
    NEBULA = 1
    ASTEROID = 2


class ActionType(IntEnum):
    """Unit action identifiers expected by the Lux runner."""

    WAIT = 0
    MOVE_UP = 1
    MOVE_RIGHT = 2
    MOVE_DOWN = 3
    MOVE_LEFT = 4
    SAP = 5


DIRECTION_DELTAS: dict[ActionType, Position] = {
    ActionType.MOVE_UP: (0, -1),
    ActionType.MOVE_RIGHT: (1, 0),
    ActionType.MOVE_DOWN: (0, 1),
    ActionType.MOVE_LEFT: (-1, 0),
}


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """The public game parameters plus conservative planning defaults."""

    map_width: int = 24
    map_height: int = 24
    max_units: int = 16
    max_steps_in_match: int = 100
    match_count_per_episode: int = 5
    unit_move_cost: int = 2
    unit_sap_cost: int = 30
    unit_sap_range: int = 4
    unit_sensor_range: int = 2
    relic_config_size: int = 5
    max_relic_nodes: int = 6
    low_energy_moves: int = 3
    unknown_tile_cost: float = 0.2
    nebula_cost: float = 0.7
    negative_energy_cost: float = 0.08
    # Drift cadence/direction are hidden from official agents. Only information
    # from the current observation is trusted for routing and energy decisions.
    terrain_ttl: int = 0
    energy_ttl: int = 0
    score_negative_ttl: int = 6

    @classmethod
    def from_env(cls, raw: Any) -> AgentConfig:
        """Build a config from either a mapping or a dataclass-like object."""

        def read(name: str, default: Any) -> Any:
            if isinstance(raw, dict):
                return raw.get(name, default)
            return getattr(raw, name, default)

        defaults = cls()
        return cls(
            map_width=int(read("map_width", defaults.map_width)),
            map_height=int(read("map_height", defaults.map_height)),
            max_units=int(read("max_units", defaults.max_units)),
            max_steps_in_match=int(
                read("max_steps_in_match", defaults.max_steps_in_match)
            ),
            match_count_per_episode=int(
                read("match_count_per_episode", defaults.match_count_per_episode)
            ),
            unit_move_cost=int(read("unit_move_cost", defaults.unit_move_cost)),
            unit_sap_cost=int(read("unit_sap_cost", defaults.unit_sap_cost)),
            unit_sap_range=int(read("unit_sap_range", defaults.unit_sap_range)),
            unit_sensor_range=int(
                read("unit_sensor_range", defaults.unit_sensor_range)
            ),
            relic_config_size=int(
                read("relic_config_size", defaults.relic_config_size)
            ),
            max_relic_nodes=int(
                read("max_relic_nodes", defaults.max_relic_nodes)
            ),
            low_energy_moves=int(
                read("low_energy_moves", defaults.low_energy_moves)
            ),
            unknown_tile_cost=float(
                read("unknown_tile_cost", defaults.unknown_tile_cost)
            ),
            nebula_cost=float(read("nebula_cost", defaults.nebula_cost)),
            negative_energy_cost=float(
                read("negative_energy_cost", defaults.negative_energy_cost)
            ),
            terrain_ttl=int(read("terrain_ttl", defaults.terrain_ttl)),
            energy_ttl=int(read("energy_ttl", defaults.energy_ttl)),
            score_negative_ttl=int(
                read("score_negative_ttl", defaults.score_negative_ttl)
            ),
        )


def mirror_position(position: Position, width: int, height: int) -> Position:
    """Mirror a coordinate using Lux S3's anti-diagonal map symmetry."""

    if width != height:
        raise ValueError("Lux anti-diagonal symmetry requires a square map")
    x, y = position
    return width - 1 - y, height - 1 - x


def in_bounds(position: Position, width: int, height: int) -> bool:
    x, y = position
    return 0 <= x < width and 0 <= y < height


def manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def chebyshev(left: Position, right: Position) -> int:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1]))
