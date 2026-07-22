import numpy as np
import pytest
from conftest import make_observation

from cortex_arena.agent import Agent
from cortex_arena.config import ActionType


@pytest.mark.parametrize(("player", "team_id"), [("player_0", 0), ("player_1", 1)])
def test_agent_emits_bounded_action_matrix(config, player, team_id) -> None:
    agent = Agent(player, config)
    obs = make_observation(
        config,
        friendly={0: ((0, 0), 100), 1: ((1, 0), 100)},
        enemies={0: ((3, 3), 20)},
        relics=[(3, 3)],
        match_step=4,
        team_id=team_id,
    )
    actions = agent.act(4, obs)
    assert actions.shape == (config.max_units, 3)
    assert actions.dtype == np.int64
    assert np.all((actions[:, 0] >= 0) & (actions[:, 0] <= 5))
    assert np.all(np.abs(actions[:, 1:]) <= config.unit_sap_range)
    assert ActionType.SAP in actions[:, 0]
