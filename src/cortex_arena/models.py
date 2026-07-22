"""Small immutable records shared by perception and planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from cortex_arena.config import Position


class Role(StrEnum):
    SCOUT = "scout"
    PROSPECTOR = "prospector"
    HARVESTER = "harvester"
    INTERCEPTOR = "interceptor"
    RECHARGE = "recharge"


@dataclass(frozen=True, slots=True)
class UnitSnapshot:
    unit_id: int
    position: Position
    energy: int


@dataclass(frozen=True, slots=True)
class OpponentTrack:
    unit_id: int
    position: Position
    energy: int
    last_seen_step: int


@dataclass(frozen=True, slots=True)
class Assignment:
    unit_id: int
    role: Role
    target: Position
    reason: str


@dataclass(frozen=True, slots=True)
class SapOrder:
    unit_id: int
    target: Position
    utility: float
