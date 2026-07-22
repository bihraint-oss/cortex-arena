"""Deterministic exploration baseline used for installation smoke tests."""

from __future__ import annotations

import json
from argparse import Namespace

import numpy as np


def direction_to(source: np.ndarray, target: np.ndarray) -> int:
    dx, dy = target - source
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 4
    return 3 if dy > 0 else 1


class StarterAgent:
    def __init__(self, player: str, config: dict) -> None:
        self.team_id = 0 if player == "player_0" else 1
        self.config = config
        self.relics: list[tuple[int, int]] = []
        width, height = int(config["map_width"]), int(config["map_height"])
        base_waypoints = [
            (3, 3),
            (3, 12),
            (12, 3),
            (9, 9),
            (6, 18),
            (18, 6),
            (15, 15),
            (20, 20),
        ]
        if self.team_id == 0:
            self.waypoints = base_waypoints
        else:
            self.waypoints = [
                (width - 1 - x, height - 1 - y) for x, y in base_waypoints
            ]

    def act(self, step: int, obs: dict) -> np.ndarray:
        mask = np.asarray(obs["units_mask"][self.team_id], dtype=bool)
        positions = np.asarray(obs["units"]["position"][self.team_id], dtype=int)
        relic_mask = np.asarray(obs["relic_nodes_mask"], dtype=bool)
        relic_positions = np.asarray(obs["relic_nodes"], dtype=int)
        for relic_id in np.flatnonzero(relic_mask):
            relic = tuple(int(value) for value in relic_positions[relic_id])
            if relic not in self.relics:
                self.relics.append(relic)

        actions = np.zeros((int(self.config["max_units"]), 3), dtype=int)
        for unit_id in np.flatnonzero(mask):
            position = positions[unit_id]
            if self.relics:
                relic = self.relics[int(unit_id) % len(self.relics)]
                offset = ((int(unit_id) % 5) - 2, ((int(unit_id) // 5) % 5) - 2)
                target = np.array([relic[0] + offset[0], relic[1] + offset[1]])
            else:
                target = np.array(
                    self.waypoints[(int(unit_id) + step // 24) % len(self.waypoints)]
                )
            actions[unit_id, 0] = direction_to(position, target)
        return actions


AGENTS: dict[str, StarterAgent] = {}


def agent_fn(observation: Namespace, config: dict) -> dict:
    obs = observation.obs
    if isinstance(obs, str):
        obs = json.loads(obs)
    if observation.step == 0 or observation.player not in AGENTS:
        AGENTS[observation.player] = StarterAgent(
            observation.player, config["env_cfg"]
        )
    return {"action": AGENTS[observation.player].act(observation.step, obs).tolist()}


def run_stdio() -> None:
    env_cfg = None
    while True:
        try:
            payload = json.loads(input())
        except EOFError:
            return
        if env_cfg is None:
            env_cfg = payload["info"]["env_cfg"]
        observation = Namespace(
            step=payload["step"],
            obs=payload["obs"],
            player=payload["player"],
        )
        print(json.dumps(agent_fn(observation, {"env_cfg": env_cfg})), flush=True)
