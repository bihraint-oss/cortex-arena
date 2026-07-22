import argparse
import hashlib
import tarfile
from pathlib import Path

import pytest

import cortex_arena.cli as cli
from cortex_arena.cli import (
    _parse_rewards,
    _runner_environment,
    _safe_package_read,
    build_parser,
    command_build_submission,
)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Rewards: {'player_0': 1.0, 'player_1': 0.0}", (1.0, 0.0)),
        (
            "Rewards: {'player_0': array(0., dtype=float32), 'player_1': array(1., dtype=float32)}",
            (0.0, 1.0),
        ),
        ("Rewards: {'player_0': np.float32(1.0), 'player_1': np.float32(0.0)}", (1.0, 0.0)),
    ],
)
def test_reward_parser_handles_runner_scalar_formats(line, expected) -> None:
    assert _parse_rewards(line) == expected


def test_parser_requires_a_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    assert parser.parse_args(["doctor"]).command == "doctor"


def test_runner_environment_prefers_current_interpreter(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")

    first = _runner_environment()["PATH"].split(":", maxsplit=1)[0]

    assert first == str(Path(cli.sys.executable).parent)


def test_submission_contains_entrypoint_and_package(tmp_path) -> None:
    output = tmp_path / "submission.tar.gz"
    result = command_build_submission(argparse.Namespace(output=str(output)))
    assert result == 0
    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
        owners = {(member.uid, member.gid, member.uname, member.gname) for member in archive}
    assert "main.py" in names
    assert "cortex_arena/agent.py" in names
    assert "LICENSE" in names
    assert owners == {(0, 0, "", "")}

    first_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    command_build_submission(argparse.Namespace(output=str(output)))
    assert hashlib.sha256(output.read_bytes()).hexdigest() == first_hash


def test_submission_reader_rejects_symlink(monkeypatch, tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    secret = tmp_path / "secret.py"
    secret.write_text("TOKEN = 'must-not-leak'\n", encoding="utf-8")
    (package / "agent.py").symlink_to(secret)
    monkeypatch.setattr(cli, "PACKAGE_ROOT", package)

    with pytest.raises(ValueError, match="symlink"):
        _safe_package_read("agent.py")


@pytest.mark.parametrize("relative", ["../secret.py", "/tmp/secret.py"])
def test_submission_reader_rejects_path_traversal(relative: str) -> None:
    with pytest.raises(ValueError, match="Unsafe package path"):
        _safe_package_read(relative)
