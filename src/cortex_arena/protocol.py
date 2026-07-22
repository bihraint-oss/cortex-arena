"""Official line-oriented Lux/Kaggle agent protocol adapter."""

from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

from cortex_arena.agent import Agent

AGENTS: dict[str, Agent] = {}


def agent_fn(observation: Any, configurations: dict) -> dict:
    """Kaggle-compatible callable used by local and hosted runners."""

    obs = observation.obs
    if isinstance(obs, str):
        obs = json.loads(obs)
    player = observation.player
    if observation.step == 0 or player not in AGENTS:
        AGENTS[player] = Agent(player, configurations["env_cfg"])
    actions = AGENTS[player].act(
        observation.step,
        obs,
        getattr(observation, "remainingOverageTime", 60),
    )
    return {"action": actions.tolist()}


def run_stdio() -> None:
    """Serve decisions over stdin/stdout until the runner closes the stream."""

    env_cfg: dict | None = None
    while True:
        try:
            line = input()
        except EOFError:
            return
        payload = json.loads(line)
        observation = Namespace(
            step=payload["step"],
            obs=payload["obs"],
            remainingOverageTime=payload.get("remainingOverageTime", 60),
            player=payload["player"],
            info=payload.get("info", {}),
        )
        if env_cfg is None:
            env_cfg = payload["info"]["env_cfg"]
        print(json.dumps(agent_fn(observation, {"env_cfg": env_cfg})), flush=True)
