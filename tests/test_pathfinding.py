
from cortex_arena.config import ActionType, TileType
from cortex_arena.pathfinding import Pathfinder, direction_to
from cortex_arena.world import WorldModel


def test_astar_routes_around_known_asteroid(config) -> None:
    world = WorldModel(0, config)
    world.tile_type[:] = TileType.EMPTY
    world.energy[:] = 0
    world.last_seen[:] = 0
    world.step = 0
    world.tile_type[1, 0] = TileType.ASTEROID
    action, next_position = Pathfinder(config).next_action(world, (0, 0), (2, 0))
    assert action == ActionType.MOVE_DOWN
    assert next_position == (0, 1)


def test_wait_and_direction_conversion(config) -> None:
    world = WorldModel(0, config)
    world.tile_type[:] = TileType.EMPTY
    world.last_seen[:] = 0
    world.step = 0
    action, position = Pathfinder(config).next_action(world, (2, 2), (2, 2))
    assert action == ActionType.WAIT
    assert position == (2, 2)
    assert direction_to((2, 2), (3, 2)) == ActionType.MOVE_RIGHT
    assert direction_to((2, 2), (4, 2)) == ActionType.WAIT


def test_stale_asteroid_is_not_a_permanent_barrier(config) -> None:
    world = WorldModel(0, config)
    world.tile_type[:] = TileType.EMPTY
    world.last_seen[:] = 0
    world.tile_type[1, 0] = TileType.ASTEROID
    world.step = config.terrain_ttl + 1

    assert world.is_passable((1, 0))
    action, _ = Pathfinder(config).next_action(world, (0, 0), (2, 0))
    assert action == ActionType.MOVE_RIGHT
