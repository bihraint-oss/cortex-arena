from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cortex_arena.config import AgentConfig


@pytest.fixture
def config() -> AgentConfig:
    return AgentConfig(
        map_width=8,
        map_height=8,
        max_units=4,
        max_steps_in_match=20,
        unit_move_cost=2,
        unit_sap_cost=30,
        unit_sap_range=4,
        unit_sensor_range=2,
    )


def make_observation(
    config: AgentConfig,
    *,
    friendly: dict[int, tuple[tuple[int, int], int]] | None = None,
    enemies: dict[int, tuple[tuple[int, int], int]] | None = None,
    relics: list[tuple[int, int]] | None = None,
    points: tuple[int, int] = (0, 0),
    team_id: int = 0,
    match_step: int = 0,
    sensor: np.ndarray | None = None,
    tile_type: np.ndarray | None = None,
    energy_map: np.ndarray | None = None,
) -> dict[str, Any]:
    friendly = friendly or {}
    enemies = enemies or {}
    relics = relics or []
    positions = np.full((2, config.max_units, 2), -1, dtype=np.int16)
    energies = np.full((2, config.max_units), -1, dtype=np.int16)
    masks = np.zeros((2, config.max_units), dtype=bool)

    for unit_id, (position, unit_energy) in friendly.items():
        positions[team_id, unit_id] = position
        energies[team_id, unit_id] = unit_energy
        masks[team_id, unit_id] = True
    for unit_id, (position, unit_energy) in enemies.items():
        opponent = 1 - team_id
        positions[opponent, unit_id] = position
        energies[opponent, unit_id] = unit_energy
        masks[opponent, unit_id] = True

    relic_positions = np.full((6, 2), -1, dtype=np.int16)
    relic_mask = np.zeros(6, dtype=bool)
    for relic_id, position in enumerate(relics):
        relic_positions[relic_id] = position
        relic_mask[relic_id] = True

    shape = (config.map_width, config.map_height)
    if sensor is None:
        sensor = np.ones(shape, dtype=bool)
    if tile_type is None:
        tile_type = np.zeros(shape, dtype=np.int8)
    if energy_map is None:
        energy_map = np.zeros(shape, dtype=np.int16)

    return {
        "units": {"position": positions, "energy": energies},
        "units_mask": masks,
        "sensor_mask": sensor,
        "map_features": {"tile_type": tile_type, "energy": energy_map},
        "relic_nodes": relic_positions,
        "relic_nodes_mask": relic_mask,
        "team_points": np.asarray(points, dtype=np.int32),
        "team_wins": np.zeros(2, dtype=np.int32),
        "steps": match_step,
        "match_steps": match_step,
    }
