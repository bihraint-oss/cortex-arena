from cortex_arena.combat import CombatPlanner
from cortex_arena.models import OpponentTrack, Role, UnitSnapshot
from cortex_arena.planner import TacticalPlanner
from cortex_arena.world import WorldModel


def test_planner_holds_confirmed_tile_and_recharges_low_energy_unit(config) -> None:
    world = WorldModel(0, config)
    world.tile_type[:] = 0
    world.energy[:] = 0
    world.last_seen[:] = 0
    world.energy_actionable[:] = True
    world.step = 0
    world.energy[1, 1] = 10
    world.score_known[3, 3] = world.SCORING
    world.candidate_mask[3, 3] = True
    units = [UnitSnapshot(0, (3, 3), 100), UnitSnapshot(1, (0, 0), 4)]

    assignments = TacticalPlanner(config).assign(world, units, points_behind=False)

    assert assignments[0].role == Role.HARVESTER
    assert assignments[0].target == (3, 3)
    assert assignments[1].role == Role.RECHARGE
    assert assignments[1].target == (1, 1)


def test_recharge_requires_fresh_positive_reachable_energy(config) -> None:
    world = WorldModel(0, config)
    world.tile_type[:] = 0
    world.energy[:] = -1
    world.last_seen[:] = 0
    world.energy_actionable[:] = True
    world.step = 0
    assert world.best_energy_tile((0, 0), available_energy=4) is None

    world.energy[1, 1] = 10
    world.energy[1, 0] = 0
    world.last_seen[1, 1] = 0
    world.last_seen[1, 0] = 0
    assert world.best_energy_tile((0, 0), available_energy=3) is None
    assert world.best_energy_tile((0, 0), available_energy=4) == (1, 1)

    world.step = config.energy_ttl + 1
    assert world.best_energy_tile((0, 0), available_energy=4) is None


def test_recharge_rejects_route_that_dies_in_negative_field(config) -> None:
    world = WorldModel(0, config)
    world.tile_type[:] = 0
    world.energy[:] = 0
    world.last_seen[:] = 0
    world.energy_actionable[:] = True
    world.step = 0
    world.energy[1, 0] = -20
    world.energy[0, 1] = -20
    world.energy[1, 1] = 10

    assert world.best_energy_tile((0, 0), available_energy=4) is None


def test_unchanged_negative_field_still_blocks_recharge_route(config) -> None:
    world = WorldModel(0, config)
    world.tile_type[:] = 0
    world.energy[:] = 0
    world.last_seen[:] = 2
    world.step = 2
    world.energy[1, 0] = -20
    world.energy[0, 1] = -20
    world.energy[2, 0] = 10
    world.energy_actionable[:] = False
    world.energy_actionable[2, 0] = True

    assert world.known_energy((1, 0)) == -20
    assert world.best_energy_tile((0, 0), available_energy=4) is None


def test_combat_planner_targets_visible_stack(config) -> None:
    world = WorldModel(0, config)
    world.tile_type[:] = 0
    world.visible_opponents = {
        0: OpponentTrack(0, (4, 4), 25, 10),
        1: OpponentTrack(1, (4, 4), 40, 10),
    }
    units = [UnitSnapshot(0, (2, 2), 100), UnitSnapshot(1, (1, 1), 100)]

    orders = CombatPlanner(config).plan(world, units, points_behind=False)

    assert orders
    assert all(order.target == (4, 4) for order in orders.values())


def test_sap_is_legal_at_exact_energy_cost(config) -> None:
    world = WorldModel(0, config)
    world.tile_type[:] = 0
    world.visible_opponents = {0: OpponentTrack(0, (3, 3), 20, 10)}
    units = [UnitSnapshot(0, (1, 1), config.unit_sap_cost)]

    orders = CombatPlanner(config).plan(world, units, points_behind=True)

    assert 0 in orders


def test_combat_ignores_enemy_already_below_zero_energy(config) -> None:
    world = WorldModel(0, config)
    world.visible_opponents = {0: OpponentTrack(0, (3, 3), -1, 10)}
    units = [UnitSnapshot(0, (1, 1), 100)]

    assert CombatPlanner(config).plan(world, units, points_behind=True) == {}


def test_exact_multiple_sap_requires_one_extra_shot(config) -> None:
    world = WorldModel(0, config)
    world.visible_opponents = {
        0: OpponentTrack(0, (3, 3), 2 * config.unit_sap_cost, 10)
    }
    units = [UnitSnapshot(unit_id, (1, unit_id + 1), 100) for unit_id in range(3)]

    orders = CombatPlanner(config).plan(world, units, points_behind=True)

    assert len(orders) == 3
