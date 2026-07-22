from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from dataclasses import asdict

import numpy as np
from conftest import make_observation

from cortex_arena.cli import command_build_submission


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def test_extracted_submission_serves_one_protocol_step(config, tmp_path) -> None:
    bundle = tmp_path / "submission.tar.gz"
    extracted = tmp_path / "submission"
    extracted.mkdir()
    command_build_submission(argparse.Namespace(output=str(bundle)))
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(extracted, filter="data")

    payload = {
        "step": 0,
        "player": "player_0",
        "remainingOverageTime": 60,
        "obs": _jsonable(
            make_observation(
                config,
                friendly={0: ((0, 0), 100)},
                relics=[(3, 3)],
                match_step=0,
            )
        ),
        "info": {"env_cfg": asdict(config)},
    }
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=extracted,
        input=json.dumps(payload) + "\n",
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    actions = np.asarray(response["action"])
    assert actions.shape == (config.max_units, 3)
