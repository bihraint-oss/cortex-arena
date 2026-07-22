"""Top-level perception → planning → action agent loop."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np

from cortex_arena.combat import CombatPlanner
from cortex_arena.config import ActionType, AgentConfig
from cortex_arena.models import Assignment
from cortex_arena.pathfinding import Pathfinder
from cortex_arena.planner import TacticalPlanner
from cortex_arena.world import WorldModel


class Agent:
    """A deterministic hierarchical controller compatible with the Lux runner."""

    def __init__(self, player: str, env_cfg: Any) -> None:
        self.player = player
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id
        self.config = AgentConfig.from_env(env_cfg)
        self.world = WorldModel(self.team_id, self.config)
        self.pathfinder = Pathfinder(self.config)
        self.planner = TacticalPlanner(self.config)
        self.combat = CombatPlanner(self.config)
        self.debug = os.getenv("CORTEX_DEBUG", "").lower() in {"1", "true", "yes"}
        self.last_assignments: dict[int, Assignment] = {}

    def act(
        self,
        step: int,
        obs: dict,
        remainingOverageTime: int = 60,
    ) -> np.ndarray:
        """Return one validated ``(max_units, 3)`` action matrix."""

        del remainingOverageTime  # The policy is bounded and never spends overtime.
        self.world.update(step, obs)
        units = list(self.world.friendly_units.values())
        team_points = np.asarray(obs["team_points"], dtype=np.int32)
        points_behind = bool(team_points[self.team_id] < team_points[self.opp_team_id])

        assignments = self.planner.assign(
            self.world, units, points_behind=points_behind
        )
        sap_orders = self.combat.plan(
            self.world, units, points_behind=points_behind
        )
        self.last_assignments = assignments

        actions = np.zeros((self.config.max_units, 3), dtype=np.int64)
        for unit in units:
            sap = sap_orders.get(unit.unit_id)
            if sap is not None:
                dx = int(sap.target[0] - unit.position[0])
                dy = int(sap.target[1] - unit.position[1])
                if (
                    abs(dx) <= self.config.unit_sap_range
                    and abs(dy) <= self.config.unit_sap_range
                ):
                    actions[unit.unit_id] = [ActionType.SAP, dx, dy]
                    continue

            assignment = assignments.get(unit.unit_id)
            if assignment is None or unit.energy < self.config.unit_move_cost:
                continue
            move, _ = self.pathfinder.next_action(
                self.world, unit.position, assignment.target
            )
            actions[unit.unit_id] = [int(move), 0, 0]

        if self.debug:
            self._trace(step, actions, sap_orders)
        return actions

    def _trace(self, step: int, actions: np.ndarray, sap_orders: dict) -> None:
        trace = {
            "step": int(step),
            "match": self.world.match_index,
            "match_step": self.world.match_step,
            "known_relics": len(self.world.relic_nodes),
            "confirmed_score_tiles": len(self.world.confirmed_scoring_tiles()),
            "likely_score_tiles": int(
                np.count_nonzero(self.world.policy_score == self.world.SCORING)
            ),
            "max_score_belief": round(
                float(np.nanmax(self.world.score_belief)), 3
            )
            if np.any(~np.isnan(self.world.score_belief))
            else None,
            "visible_opponents": len(self.world.visible_opponents),
            "saps": len(sap_orders),
            "roles": {
                str(unit_id): assignment.role.value
                for unit_id, assignment in self.last_assignments.items()
            },
            "non_idle_actions": int(np.count_nonzero(actions[:, 0])),
        }
        print(f"CORTEX_TRACE {json.dumps(trace, sort_keys=True)}", file=sys.stderr)
