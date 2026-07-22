"""Developer CLI for replays, benchmarks, diagnostics, and submissions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import time
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cortex_arena import __version__

PACKAGE_ROOT = Path(__file__).resolve().parent
CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
IN_CHECKOUT = (CHECKOUT_ROOT / "main.py").is_file()
RUN_ROOT = CHECKOUT_ROOT if IN_CHECKOUT else Path.cwd()
MAIN_AGENT = (
    CHECKOUT_ROOT / "main.py"
    if IN_CHECKOUT
    else PACKAGE_ROOT / "player_scripts" / "cortex.py"
)
STARTER_AGENT = (
    CHECKOUT_ROOT / "opponents" / "starter" / "main.py"
    if IN_CHECKOUT
    else PACKAGE_ROOT / "player_scripts" / "starter.py"
)

EXPECTED_LUX_VERSION = "0.2.1"
SUBMISSION_RUNTIME_FILES = (
    "__init__.py",
    "agent.py",
    "combat.py",
    "config.py",
    "models.py",
    "pathfinding.py",
    "planner.py",
    "protocol.py",
    "world.py",
)


def _runner_environment() -> dict[str, str]:
    """Ensure Lux launches child agents with this installation's Python."""

    environment = os.environ.copy()
    # Do not resolve the interpreter symlink: its parent is the virtualenv's
    # ``bin`` directory, while the resolved target belongs to the base runtime.
    interpreter_directory = str(Path(sys.executable).parent)
    existing_path = environment.get("PATH", "")
    environment["PATH"] = os.pathsep.join(
        part for part in (interpreter_directory, existing_path) if part
    )
    return environment


def _runner_command(
    first: Path,
    second: Path,
    *,
    seed: int,
    output: Path | None = None,
    render: bool = False,
    verbose: int = 1,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "luxai_runner.cli",
        str(first),
        str(second),
        f"--seed={seed}",
        f"--verbose={verbose}",
    ]
    if output is not None:
        command.append(f"--output={output}")
    if render:
        command.append("--render")
    return command


def _parse_rewards(output: str) -> tuple[float, float]:
    reward_line = next(
        (line for line in reversed(output.splitlines()) if "Rewards:" in line),
        "",
    )
    if not reward_line:
        raise ValueError("Lux runner output did not contain final rewards")

    values: list[float] = []
    for player in ("player_0", "player_1"):
        match = re.search(
            rf"['\"]?{player}['\"]?\s*:\s*"
            rf"(?:(?:array|Array|float\d+|np\.float\d+)\()?\s*"
            rf"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)",
            reward_line,
        )
        if match is None:
            raise ValueError(f"Could not parse {player} reward from: {reward_line}")
        values.append(float(match.group(1)))
    return values[0], values[1]


def command_play(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = _runner_command(
        MAIN_AGENT,
        STARTER_AGENT,
        seed=args.seed,
        output=output,
        render=args.render,
        verbose=args.verbose,
    )
    print(f"Running Cortex Arena (seed={args.seed})")
    result = subprocess.run(
        command,
        cwd=RUN_ROOT,
        env=_runner_environment(),
        check=False,
    )
    if result.returncode == 0:
        print(f"Replay written to {output}")
    return int(result.returncode)


def command_benchmark(args: argparse.Namespace) -> int:
    wins = losses = draws = 0
    episodes: list[dict] = []
    started = time.perf_counter()
    for index in range(args.games):
        seed = args.seed + index
        cortex_first = index % 2 == 0
        first, second = (
            (MAIN_AGENT, STARTER_AGENT)
            if cortex_first
            else (STARTER_AGENT, MAIN_AGENT)
        )
        result = subprocess.run(
            _runner_command(first, second, seed=seed, verbose=0),
            cwd=RUN_ROOT,
            env=_runner_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return int(result.returncode)
        reward_0, reward_1 = _parse_rewards(result.stdout)
        cortex_reward = reward_0 if cortex_first else reward_1
        opponent_reward = reward_1 if cortex_first else reward_0
        if cortex_reward > opponent_reward:
            outcome = "win"
            wins += 1
        elif cortex_reward < opponent_reward:
            outcome = "loss"
            losses += 1
        else:
            outcome = "draw"
            draws += 1
        episodes.append(
            {
                "seed": seed,
                "side": "player_0" if cortex_first else "player_1",
                "outcome": outcome,
                "reward": cortex_reward,
                "opponent_reward": opponent_reward,
            }
        )
        print(f"[{index + 1:02d}/{args.games:02d}] seed={seed} {outcome}")

    elapsed = time.perf_counter() - started
    try:
        environment_version = version("luxai-s3")
    except PackageNotFoundError:  # pragma: no cover - doctor catches this first
        environment_version = "unknown"
    summary = {
        "agent": "cortex-arena",
        "version": __version__,
        "environment": f"luxai-s3=={environment_version}",
        "agent_source_sha256": _source_hash(
            [PACKAGE_ROOT / name for name in SUBMISSION_RUNTIME_FILES]
            + [MAIN_AGENT]
        ),
        "opponent": "bundled-starter",
        "opponent_source_sha256": _source_hash(
            [PACKAGE_ROOT / "starter.py", STARTER_AGENT]
        ),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "games": args.games,
        "seed_start": args.seed,
        "seed_end": args.seed + args.games - 1 if args.games else args.seed,
        "side_policy": "alternate",
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / args.games if args.games else 0.0,
        "match_wins": sum(item["reward"] for item in episodes),
        "opponent_match_wins": sum(item["opponent_reward"] for item in episodes),
        "elapsed_seconds": round(elapsed, 3),
        "episodes": episodes,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Benchmark report written to {destination}")
    return 0


def command_doctor(_: argparse.Namespace) -> int:
    python_supported = (3, 11) <= sys.version_info[:2] < (3, 14)
    checks: list[tuple[str, str, bool]] = [
        ("Cortex Arena", __version__, True),
        ("Python", platform.python_version(), python_supported),
        ("Platform", f"{platform.system()} {platform.machine()}", True),
        ("Main agent", str(MAIN_AGENT), MAIN_AGENT.is_file()),
        ("Starter opponent", str(STARTER_AGENT), STARTER_AGENT.is_file()),
    ]
    try:
        lux_version = version("luxai-s3")
        checks.append(
            (
                "Lux AI S3",
                lux_version,
                lux_version == EXPECTED_LUX_VERSION,
            )
        )
    except PackageNotFoundError as exc:  # pragma: no cover - broken installation
        checks.append(("Lux AI S3", str(exc), False))

    for name, detail, passed in checks:
        print(f"{'OK' if passed else 'FAIL':4}  {name:18} {detail}")
    return 0 if all(passed for _, _, passed in checks) else 1


def command_build_submission(args: argparse.Namespace) -> int:
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    members: list[tuple[str, bytes]] = [
        (
            "main.py",
            _safe_package_read("player_scripts/cortex.py"),
        )
    ]
    for relative in SUBMISSION_RUNTIME_FILES:
        arcname = Path("cortex_arena") / relative
        members.append((arcname.as_posix(), _safe_package_read(relative)))
    members.extend(
        [
            ("LICENSE", _safe_package_read("LICENSE.txt")),
            ("NOTICE", _safe_package_read("NOTICE.txt")),
        ]
    )
    _write_deterministic_tar(destination, members)
    print(f"Submission bundle written to {destination}")
    return 0


def _safe_package_read(relative: str) -> bytes:
    """Read an allowlisted package member without traversing a symlink."""

    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe package path: {relative}")

    root = PACKAGE_ROOT.resolve(strict=True)
    source = PACKAGE_ROOT / relative_path
    current = source
    while current != PACKAGE_ROOT:
        if current.is_symlink():
            raise ValueError(f"Submission source must not be a symlink: {relative}")
        current = current.parent

    try:
        resolved = source.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"Submission source escapes the package: {relative}") from exc
    if not resolved.is_file():
        raise ValueError(f"Submission source is not a file: {relative}")
    return source.read_bytes()


def _source_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for source in sorted(paths, key=lambda item: item.name):
        digest.update(source.name.encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_deterministic_tar(
    destination: Path, members: Sequence[tuple[str, bytes]]
) -> None:
    """Write a byte-reproducible gzip/tar without local owner metadata."""

    with destination.open("wb") as raw_file, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw_file, compresslevel=9, mtime=0
    ) as gzip_file, tarfile.open(
        fileobj=gzip_file, mode="w", format=tarfile.PAX_FORMAT
    ) as archive:
        for arcname, data in sorted(members):
            info = tarfile.TarInfo(arcname)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(data))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortex-arena",
        description="Run and evaluate the Cortex Arena strategy agent.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="verify the local installation")
    doctor.set_defaults(handler=command_doctor)

    play = subparsers.add_parser("play", help="play against the bundled baseline")
    play.add_argument("--seed", type=int, default=42)
    play.add_argument("--output", default="replays/cortex-vs-starter.html")
    play.add_argument("--render", action="store_true", help="also open the live renderer")
    play.add_argument("--verbose", type=int, choices=range(4), default=1)
    play.set_defaults(handler=command_play)

    benchmark = subparsers.add_parser(
        "benchmark", help="run seeded matches with alternating sides"
    )
    benchmark.add_argument("--games", type=int, default=10)
    benchmark.add_argument("--seed", type=int, default=100)
    benchmark.add_argument("--output", default="benchmark-results.json")
    benchmark.set_defaults(handler=command_benchmark)

    submission = subparsers.add_parser(
        "build-submission", help="create a Kaggle-compatible tarball"
    )
    submission.add_argument("--output", default="dist/cortex-arena-submission.tar.gz")
    submission.set_defaults(handler=command_build_submission)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
