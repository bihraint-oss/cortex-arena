import numpy as np
from conftest import make_observation

from cortex_arena.config import TileType
from cortex_arena.world import WorldModel


def test_visible_map_information_is_mirrored(config) -> None:
    world = WorldModel(0, config)
    sensor = np.zeros((8, 8), dtype=bool)
    sensor[1, 2] = True
    tiles = np.full((8, 8), -1, dtype=np.int8)
    energy = np.full((8, 8), -1, dtype=np.int16)
    tiles[1, 2] = TileType.ASTEROID
    energy[1, 2] = 7
    obs = make_observation(config, sensor=sensor, tile_type=tiles, energy_map=energy)

    world.update(0, obs)

    assert world.tile_type[1, 2] == TileType.ASTEROID
    assert world.tile_type[5, 6] == TileType.ASTEROID
    assert world.energy[5, 6] == 7
    assert not world.is_passable((1, 2))


def test_point_delta_identifies_scoring_tile_and_its_mirror(config) -> None:
    world = WorldModel(0, config)
    position = (3, 3)
    first = make_observation(
        config,
        friendly={0: (position, 100)},
        relics=[position],
        points=(0, 0),
        match_step=1,
    )
    second = make_observation(
        config,
        friendly={0: (position, 100)},
        relics=[position],
        points=(1, 0),
        match_step=2,
    )

    world.update(1, first)
    world.update(2, second)

    mirror = (4, 4)
    assert world.score_known[position] == world.SCORING
    assert world.score_known[mirror] == world.SCORING
    assert position in world.confirmed_scoring_tiles()


def test_point_delta_can_reveal_scoring_before_relic_is_visible(config) -> None:
    world = WorldModel(0, config)
    position = (3, 3)
    world.update(
        1,
        make_observation(
            config, friendly={0: (position, 100)}, points=(0, 0), match_step=1
        ),
    )
    world.update(
        2,
        make_observation(
            config, friendly={0: (position, 100)}, points=(1, 0), match_step=2
        ),
    )

    assert world.candidate_mask[position]
    assert world.score_known[position] == world.SCORING


def test_candidate_and_unseen_relic_position_remain_jointly_ambiguous(config) -> None:
    world = WorldModel(0, config)
    candidate = (1, 1)
    outside = (3, 4)
    units = {0: (candidate, 100), 1: (outside, 100)}
    world.update(
        1,
        make_observation(
            config, friendly=units, relics=[(0, 0)], points=(0, 0), match_step=1
        ),
    )
    world.update(
        2,
        make_observation(
            config, friendly=units, relics=[(0, 0)], points=(1, 0), match_step=2
        ),
    )

    assert world.score_known[candidate] == world.UNKNOWN_SCORE
    assert world.score_known[outside] == world.UNKNOWN_SCORE
    assert world.candidate_mask[outside]
    assert world.policy_score[candidate] == world.SCORING

    world.update(
        3,
        make_observation(
            config, friendly=units, relics=[(0, 0)], points=(1, 0), match_step=3
        ),
    )
    assert world.policy_score[candidate] != world.SCORING


def test_negative_energy_unit_does_not_create_scoring_evidence(config) -> None:
    world = WorldModel(0, config)
    position = (3, 3)
    for step in (1, 2):
        world.update(
            step,
            make_observation(
                config,
                friendly={0: (position, -1)},
                relics=[position],
                points=(0, 0),
                match_step=step,
            ),
        )

    assert world.score_known[position] == world.UNKNOWN_SCORE


def test_zero_point_delta_rejects_distinct_candidate_tiles(config) -> None:
    world = WorldModel(0, config)
    positions = {0: ((2, 3), 100), 1: ((3, 2), 100)}
    world.update(
        1,
        make_observation(
            config, friendly=positions, relics=[(3, 3)], points=(0, 0), match_step=1
        ),
    )
    world.update(
        2,
        make_observation(
            config, friendly=positions, relics=[(3, 3)], points=(0, 0), match_step=2
        ),
    )
    assert world.score_known[2, 3] == world.NON_SCORING
    assert world.score_known[3, 2] == world.NON_SCORING


def test_new_relic_reopens_non_scoring_overlap(config) -> None:
    world = WorldModel(0, config)
    overlap = (3, 3)
    world.update(
        1,
        make_observation(
            config,
            friendly={0: (overlap, 100)},
            relics=[(1, 1)],
            points=(0, 0),
            match_step=1,
        ),
    )
    world.update(
        2,
        make_observation(
            config,
            friendly={0: (overlap, 100)},
            relics=[(1, 1)],
            points=(0, 0),
            match_step=2,
        ),
    )
    assert world.score_known[overlap] == world.NON_SCORING

    # A newly spawned relic has a different hidden scoring mask. Its 5x5
    # footprint overlaps the old negative tile and the next point delta proves it.
    world.update(
        3,
        make_observation(
            config,
            friendly={0: (overlap, 100)},
            relics=[(1, 1), (4, 4)],
            points=(1, 0),
            match_step=3,
        ),
    )
    assert world.score_known[overlap] == world.SCORING


def test_old_negative_reopens_during_unseen_relic_spawn_window(config) -> None:
    world = WorldModel(0, config)
    position = (3, 3)
    for step in (1, 2):
        world.update(
            step,
            make_observation(
                config,
                friendly={0: (position, 100)},
                relics=[(1, 1)],
                points=(0, 0),
                match_step=step,
            ),
        )
    assert world.score_known[position] == world.NON_SCORING

    later = 2 + config.score_negative_ttl + 1
    world.update(
        later,
        make_observation(
            config, relics=[(1, 1)], points=(0, 0), match_step=later
        ),
    )

    assert world.score_known[position] == world.UNKNOWN_SCORE


def test_recent_negative_can_reactivate_before_new_relic_is_seen(config) -> None:
    world = WorldModel(0, config)
    negative = (2, 3)
    other = (3, 2)
    world.update(
        1,
        make_observation(
            config,
            friendly={0: (negative, 100)},
            relics=[(3, 3)],
            points=(0, 0),
            match_step=1,
        ),
    )
    world.update(
        2,
        make_observation(
            config,
            friendly={0: (negative, 100)},
            relics=[(3, 3)],
            points=(0, 0),
            match_step=2,
        ),
    )
    assert world.score_known[negative] == world.NON_SCORING

    world.update(
        3,
        make_observation(
            config,
            friendly={0: (negative, 100), 1: (other, 100)},
            relics=[(3, 3)],
            points=(1, 0),
            match_step=3,
        ),
    )

    assert world.score_known[negative] != world.SCORING
    assert world.score_known[other] != world.SCORING


def test_duplicate_units_on_one_scoring_tile_count_once(config) -> None:
    world = WorldModel(0, config)
    position = (3, 3)
    units = {0: (position, 100), 1: (position, 100)}
    world.update(
        1,
        make_observation(
            config, friendly=units, relics=[position], points=(0, 0), match_step=1
        ),
    )
    world.update(
        2,
        make_observation(
            config, friendly=units, relics=[position], points=(1, 0), match_step=2
        ),
    )
    assert world.score_known[position] == world.SCORING


def test_partial_score_delta_updates_belief_without_false_certainty(config) -> None:
    world = WorldModel(0, config)
    positions = {0: ((2, 3), 100), 1: ((3, 2), 100)}
    world.update(
        1,
        make_observation(
            config, friendly=positions, relics=[(3, 3)], points=(0, 0), match_step=1
        ),
    )
    world.update(
        2,
        make_observation(
            config, friendly=positions, relics=[(3, 3)], points=(1, 0), match_step=2
        ),
    )
    for position, _ in positions.values():
        assert world.score_known[position] == world.UNKNOWN_SCORE
        assert 0.2 < world.score_belief[position] < 1.0


def test_match_reset_preserves_map_but_clears_point_baseline(config) -> None:
    world = WorldModel(0, config)
    world.update(
        19,
        make_observation(
            config,
            friendly={0: ((2, 2), 100)},
            relics=[(3, 3)],
            points=(7, 3),
            match_step=19,
        ),
    )
    assert world.last_team_points == 7
    observed_tile = int(world.tile_type[0, 0])

    world.update(
        21,
        make_observation(config, points=(0, 0), match_step=0),
    )
    assert world.match_index == 1
    assert world.last_team_points == 0
    assert int(world.tile_type[0, 0]) == observed_tile


def test_frontier_and_energy_queries_return_passable_targets(config) -> None:
    world = WorldModel(0, config)
    sensor = np.zeros((8, 8), dtype=bool)
    sensor[:2, :2] = True
    energy = np.full((8, 8), -1, dtype=np.int16)
    energy[:2, :2] = 0
    energy[1, 1] = 12
    tiles = np.full((8, 8), -1, dtype=np.int8)
    tiles[:2, :2] = 0
    previous_energy = energy.copy()
    previous_energy[1, 1] = 0
    world.update(
        4,
        make_observation(
            config, sensor=sensor, tile_type=tiles, energy_map=previous_energy
        ),
    )
    world.update(
        5,
        make_observation(config, sensor=sensor, tile_type=tiles, energy_map=energy),
    )
    assert world.best_energy_tile((0, 0)) == (1, 1)
    assert world.frontier_target((0, 0)) != (0, 0)


def test_hidden_terrain_drift_makes_previous_observation_unknown(config) -> None:
    world = WorldModel(0, config)
    sensor = np.zeros((8, 8), dtype=bool)
    sensor[1, 2] = True
    tiles = np.full((8, 8), -1, dtype=np.int8)
    energy = np.full((8, 8), -1, dtype=np.int16)
    tiles[1, 2] = TileType.ASTEROID
    energy[1, 2] = 7
    world.update(
        9,
        make_observation(
            config, sensor=sensor, tile_type=tiles, energy_map=energy
        ),
    )

    world.update(
        10,
        make_observation(config, sensor=np.zeros((8, 8), dtype=bool)),
    )

    assert world.current_tile_type((1, 2)) == -1
    assert world.is_passable((1, 2))


def test_hidden_energy_drift_makes_previous_observation_unknown(config) -> None:
    world = WorldModel(0, config)
    sensor = np.zeros((8, 8), dtype=bool)
    sensor[1, 1] = True
    energy = np.full((8, 8), -1, dtype=np.int16)
    energy[1, 1] = 12
    previous_energy = energy.copy()
    previous_energy[1, 1] = 0
    world.update(
        18,
        make_observation(config, sensor=sensor, energy_map=previous_energy),
    )
    world.update(
        19,
        make_observation(config, sensor=sensor, energy_map=energy),
    )
    assert world.known_energy((1, 1)) == 12

    world.update(
        20,
        make_observation(config, sensor=np.zeros((8, 8), dtype=bool)),
    )

    assert world.known_energy((1, 1)) == 0


def test_unchanged_energy_observation_is_not_used_for_recharge(config) -> None:
    world = WorldModel(0, config)
    sensor = np.zeros((8, 8), dtype=bool)
    sensor[1, 1] = True
    energy = np.full((8, 8), -1, dtype=np.int16)
    energy[1, 1] = 12
    for step in (1, 2):
        world.update(
            step,
            make_observation(config, sensor=sensor, energy_map=energy),
        )

    assert world.known_energy((1, 1)) == 0
    assert world.best_energy_tile((0, 0), available_energy=10) is None
