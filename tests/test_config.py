import pytest

from cortex_arena.config import AgentConfig, chebyshev, manhattan, mirror_position


def test_config_reads_public_environment_values() -> None:
    config = AgentConfig.from_env(
        {
            "map_width": 32,
            "map_height": 24,
            "max_units": 12,
            "unit_move_cost": 5,
            "unit_sap_cost": 41,
            "unit_sap_range": 7,
            "low_energy_moves": 7,
            "unknown_tile_cost": 1.2,
            "terrain_ttl": 9,
        }
    )
    assert config.map_width == 32
    assert config.max_units == 12
    assert config.unit_move_cost == 5
    assert config.unit_sap_cost == 41
    assert config.unit_sap_range == 7
    assert config.low_energy_moves == 7
    assert config.unknown_tile_cost == 1.2
    assert config.terrain_ttl == 9


def test_lux_anti_diagonal_symmetry() -> None:
    assert mirror_position((1, 2), 8, 8) == (5, 6)
    assert mirror_position(mirror_position((1, 2), 8, 8), 8, 8) == (1, 2)
    assert manhattan((0, 0), (2, 3)) == 5
    assert chebyshev((0, 0), (2, 3)) == 3
    with pytest.raises(ValueError, match="square map"):
        mirror_position((1, 2), 32, 24)
