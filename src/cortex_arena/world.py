"""Persistent fog-of-war memory and hidden scoring-tile inference."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np

from cortex_arena.config import (
    AgentConfig,
    Position,
    TileType,
    in_bounds,
    manhattan,
    mirror_position,
)
from cortex_arena.models import OpponentTrack, UnitSnapshot


class WorldModel:
    """Accumulates observations across all five matches in one Lux episode.

    Lux deliberately hides terrain, opponents, relics, and the exact tiles that
    score around each relic.  This model turns the observation stream into a
    stable, queryable belief state.  Knowledge persists between matches because
    the underlying map and hidden game parameters also persist.
    """

    UNKNOWN_SCORE = -1
    NON_SCORING = 0
    SCORING = 1

    def __init__(self, team_id: int, config: AgentConfig) -> None:
        self.team_id = team_id
        self.opp_team_id = 1 - team_id
        self.config = config
        shape = (config.map_width, config.map_height)

        self.tile_type = np.full(shape, -1, dtype=np.int8)
        self.energy = np.full(shape, np.nan, dtype=np.float32)
        self.energy_actionable = np.zeros(shape, dtype=np.bool_)
        self.last_seen = np.full(shape, -1, dtype=np.int32)
        self.visits = np.zeros(shape, dtype=np.int32)
        self.candidate_mask = np.zeros(shape, dtype=np.bool_)
        self.score_known = np.full(shape, self.UNKNOWN_SCORE, dtype=np.int8)
        self.policy_score = np.full(shape, self.UNKNOWN_SCORE, dtype=np.int8)
        self.score_belief = np.full(shape, np.nan, dtype=np.float32)
        self.score_last_verified = np.full(shape, -1, dtype=np.int32)

        self.relic_nodes: set[Position] = set()
        self.opponents: dict[int, OpponentTrack] = {}
        self.friendly_units: dict[int, UnitSnapshot] = {}
        self.visible_opponents: dict[int, OpponentTrack] = {}

        self.step = -1
        self.match_step = -1
        self.match_index = 0
        self.last_team_points: int | None = None
        self.last_match_step: int | None = None
        self.terrain_version = 0

    def update(self, step: int, obs: dict) -> None:
        """Merge one partial observation into the persistent world model."""

        self.step = int(step)
        match_step = int(obs.get("match_steps", step % (self.config.max_steps_in_match + 1)))
        is_new_match = (
            self.last_match_step is not None
            and match_step < self.last_match_step
        )
        if is_new_match:
            self.match_index += 1
            self.last_team_points = None
        self.match_step = match_step

        self._update_map(obs)
        self._update_relics(obs)
        self._expire_unseen_relic_negatives()
        self._update_units(obs)
        self._infer_scoring_tiles(obs, skip=is_new_match)

        self.last_match_step = match_step

    def _update_map(self, obs: dict) -> None:
        sensor = np.asarray(obs["sensor_mask"], dtype=np.bool_)
        observed_tiles = np.asarray(obs["map_features"]["tile_type"], dtype=np.int16)
        observed_energy = np.asarray(obs["map_features"]["energy"], dtype=np.float32)

        changed = False
        self.energy_actionable.fill(False)
        refreshed_energy_targets: set[Position] = set()
        for raw_x, raw_y in np.argwhere(sensor):
            position = int(raw_x), int(raw_y)
            tile = int(observed_tiles[position])
            energy = float(observed_energy[position])
            for target in self._symmetric_positions(position):
                energy_changed = (
                    self.last_seen[target] == self.step - 1
                    and not np.isnan(self.energy[target])
                    and self.energy[target] != energy
                )
                if energy_changed:
                    refreshed_energy_targets.add(target)
                if self.tile_type[target] != tile:
                    changed = True
                self.tile_type[target] = tile
                self.energy[target] = energy
                self.last_seen[target] = self.step
        for target in refreshed_energy_targets:
            self.energy_actionable[target] = True
        if changed:
            self.terrain_version += 1

    def _update_relics(self, obs: dict) -> None:
        positions = np.asarray(obs["relic_nodes"], dtype=np.int16)
        mask = np.asarray(obs["relic_nodes_mask"], dtype=np.bool_)
        newly_discovered: set[Position] = set()
        for relic_id in np.flatnonzero(mask):
            raw = positions[int(relic_id)]
            position = int(raw[0]), int(raw[1])
            if in_bounds(position, self.config.map_width, self.config.map_height):
                pair = self._symmetric_positions(position)
                newly_discovered.update(pair - self.relic_nodes)
                self.relic_nodes.update(pair)
        if newly_discovered:
            self._refresh_candidate_mask(newly_discovered)

    def _expire_unseen_relic_negatives(self) -> None:
        """Re-probe old negatives while an unseen relic pair may have spawned."""

        possible_spawn_window = (
            self.match_index < self.config.max_relic_nodes // 2
            and self.match_step <= self.config.max_steps_in_match // 2
            and len(self.relic_nodes) < self.config.max_relic_nodes
        )
        if not possible_spawn_window:
            return
        stale = (
            (self.score_known == self.NON_SCORING)
            & (self.score_last_verified >= 0)
            & (
                self.step - self.score_last_verified
                > self.config.score_negative_ttl
            )
        )
        self.score_known[stale] = self.UNKNOWN_SCORE
        self.policy_score[stale] = self.UNKNOWN_SCORE
        self.score_belief[stale] = 0.2
        self.score_last_verified[stale] = -1

    def _update_units(self, obs: dict) -> None:
        positions = np.asarray(obs["units"]["position"], dtype=np.int16)
        energies = np.asarray(obs["units"]["energy"], dtype=np.int16)
        masks = np.asarray(obs["units_mask"], dtype=np.bool_)

        self.friendly_units = {}
        for unit_id in np.flatnonzero(masks[self.team_id]):
            raw_pos = positions[self.team_id, unit_id]
            position = int(raw_pos[0]), int(raw_pos[1])
            energy = int(np.asarray(energies[self.team_id, unit_id]).item())
            snapshot = UnitSnapshot(int(unit_id), position, energy)
            self.friendly_units[int(unit_id)] = snapshot
            if in_bounds(position, self.config.map_width, self.config.map_height):
                self.visits[position] += 1

        self.visible_opponents = {}
        for unit_id in np.flatnonzero(masks[self.opp_team_id]):
            raw_pos = positions[self.opp_team_id, unit_id]
            position = int(raw_pos[0]), int(raw_pos[1])
            energy = int(np.asarray(energies[self.opp_team_id, unit_id]).item())
            track = OpponentTrack(int(unit_id), position, energy, self.step)
            self.opponents[int(unit_id)] = track
            self.visible_opponents[int(unit_id)] = track

    def _infer_scoring_tiles(self, obs: dict, *, skip: bool) -> None:
        points = int(np.asarray(obs["team_points"])[self.team_id])
        if skip or self.last_team_points is None or points < self.last_team_points:
            self.last_team_points = points
            return

        delta = points - self.last_team_points
        self.last_team_points = points
        occupied = {
            unit.position
            for unit in self.friendly_units.values()
            if unit.energy >= 0
        }
        if not occupied or delta > len(occupied):
            return

        self._infer_policy_scores(delta, occupied)

        known_positive = {p for p in occupied if self.score_known[p] == self.SCORING}
        unknown_candidates = sorted(
            p
            for p in occupied
            if self.candidate_mask[p]
            and self.score_known[p] == self.UNKNOWN_SCORE
        )
        reactivatable_negatives = sorted(
            p
            for p in occupied
            if self.candidate_mask[p]
            and self.score_known[p] == self.NON_SCORING
            and len(self.relic_nodes) < self.config.max_relic_nodes
        )
        candidate_variables = unknown_candidates + reactivatable_negatives
        outside = sorted(p for p in occupied if not self.candidate_mask[p])
        residual = delta - len(known_positive)
        unresolved = candidate_variables + outside
        if residual < 0 or residual > len(unresolved):
            return

        if residual == 0:
            for position in candidate_variables:
                self._set_score_knowledge(position, self.NON_SCORING, 0.0)
            return

        # When every occupied position is inside a known relic footprint, the
        # candidate equation is closed and can yield definitive conclusions.
        if not outside:
            if residual == len(candidate_variables):
                for position in candidate_variables:
                    self._set_score_knowledge(position, self.SCORING, 1.0)
            else:
                self._update_score_beliefs(candidate_variables, residual)
            return

        # Outside occupants may be scoring through a spawned but unseen relic.
        # Keep the joint equation ambiguous unless every unresolved tile must
        # share the same value; never force known candidates to explain it first.
        for position in outside:
            self._mark_possible_candidate(position)
        if residual == len(unresolved):
            for position in unresolved:
                self._mark_possible_candidate(position)
                self._set_score_knowledge(position, self.SCORING, 1.0)
        else:
            self._update_score_beliefs(unresolved, residual)

    def _infer_policy_scores(
        self, delta: int, occupied: set[Position]
    ) -> None:
        """Update a reversible known-relic hypothesis for exploitation.

        Unlike ``score_known``, this deliberately conditions on observed relic
        footprints explaining the points. It can drive play, but it is never
        presented as mathematical certainty and is revoked on contradiction.
        """

        policy_occupied = {p for p in occupied if self.candidate_mask[p]}
        policy_positive = {
            p for p in policy_occupied if self.policy_score[p] == self.SCORING
        }
        residual = delta - len(policy_positive)
        if residual < 0:
            for position in policy_positive:
                self._set_policy_score(position, self.UNKNOWN_SCORE)
            policy_positive.clear()
            residual = delta

        policy_unknown = sorted(
            p
            for p in policy_occupied - policy_positive
            if self.policy_score[p] == self.UNKNOWN_SCORE
        )
        if not policy_unknown or residual < 0 or residual > len(policy_unknown):
            return
        if residual == 0:
            for position in policy_unknown:
                self._set_policy_score(position, self.NON_SCORING)
        elif residual == len(policy_unknown):
            for position in policy_unknown:
                self._set_policy_score(position, self.SCORING)

    def _set_policy_score(self, position: Position, status: int) -> None:
        for target in self._symmetric_positions(position):
            self.policy_score[target] = status

    def _update_score_beliefs(
        self, positions: Iterable[Position], positive_count: int
    ) -> None:
        positions = list(positions)
        if not positions:
            return
        evidence = positive_count / len(positions)
        for position in positions:
            prior = float(self.score_belief[position])
            if np.isnan(prior):
                prior = 0.2
            posterior = 0.65 * prior + 0.35 * evidence
            for target in self._symmetric_positions(position):
                self.score_belief[target] = posterior

    def _mark_possible_candidate(self, position: Position) -> None:
        for target in self._symmetric_positions(position):
            self.candidate_mask[target] = True
            if np.isnan(self.score_belief[target]):
                self.score_belief[target] = 0.2

    def _refresh_candidate_mask(self, relics: Iterable[Position]) -> None:
        radius = self.config.relic_config_size // 2
        for relic_x, relic_y in relics:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    position = relic_x + dx, relic_y + dy
                    if not in_bounds(
                        position, self.config.map_width, self.config.map_height
                    ):
                        continue
                    self.candidate_mask[position] = True
                    # A later relic has its own hidden 5x5 mask. Its overlap can
                    # activate a tile that an earlier relic proved non-scoring,
                    # so negative knowledge in the new footprint must be reopened.
                    if self.score_known[position] == self.NON_SCORING:
                        self.score_known[position] = self.UNKNOWN_SCORE
                        self.score_belief[position] = 0.2
                        self.score_last_verified[position] = -1
                    if self.policy_score[position] == self.NON_SCORING:
                        self.policy_score[position] = self.UNKNOWN_SCORE
                    elif np.isnan(self.score_belief[position]):
                        self.score_belief[position] = 0.2

    def _set_score_knowledge(
        self, position: Position, status: int, confidence: float
    ) -> None:
        for target in self._symmetric_positions(position):
            self.score_known[target] = status
            self.score_belief[target] = confidence
            self.score_last_verified[target] = self.step

    def _symmetric_positions(self, position: Position) -> set[Position]:
        mirrored = mirror_position(
            position, self.config.map_width, self.config.map_height
        )
        return {position, mirrored}

    def is_passable(self, position: Position) -> bool:
        if not in_bounds(position, self.config.map_width, self.config.map_height):
            return False
        tile = self.current_tile_type(position)
        return tile != TileType.ASTEROID

    def current_tile_type(self, position: Position) -> int:
        """Return terrain only while it is safe under hidden global drift."""

        if self.last_seen[position] < 0:
            return -1
        if self.step - int(self.last_seen[position]) > self.config.terrain_ttl:
            return -1
        return int(self.tile_type[position])

    def known_energy(self, position: Position) -> float:
        seen = int(self.last_seen[position])
        if seen < 0 or self.step - seen > self.config.energy_ttl:
            return 0.0
        value = float(self.energy[position])
        if np.isnan(value):
            return 0.0
        # A delayed positive field is not a dependable gain, but a currently
        # observed negative value remains a conservative hazard lower bound.
        if value > 0 and not self.energy_actionable[position]:
            return 0.0
        return value

    def confirmed_scoring_tiles(self) -> list[Position]:
        return self._positions_where(self.score_known == self.SCORING)

    def policy_scoring_tiles(self) -> list[Position]:
        """Return reversible exploitation hypotheses, not proven facts."""

        return self._positions_where(self.policy_score == self.SCORING)

    def possible_scoring_tiles(self) -> list[Position]:
        mask = self.candidate_mask & (self.score_known != self.NON_SCORING)
        return self._positions_where(mask)

    def unknown_scoring_tiles(self) -> list[Position]:
        mask = self.candidate_mask & (self.score_known == self.UNKNOWN_SCORE)
        return self._positions_where(mask)

    @staticmethod
    def _positions_where(mask: np.ndarray) -> list[Position]:
        return [(int(x), int(y)) for x, y in np.argwhere(mask)]

    def best_energy_tile(
        self, origin: Position, *, available_energy: int | None = None
    ) -> Position | None:
        """Return a known, passable recharge tile with travel cost included."""

        best: Position | None = None
        best_utility = float("-inf")
        fresh = (self.last_seen >= 0) & (
            self.step - self.last_seen <= self.config.energy_ttl
        )
        known = np.argwhere(
            (~np.isnan(self.energy))
            & fresh
            & self.energy_actionable
            & (self.energy > 0)
        )
        distances = self._reachable_distances(
            origin, available_energy=available_energy
        )
        for raw_x, raw_y in known:
            position = int(raw_x), int(raw_y)
            # Nebula reduction is hidden and can exceed the visible field value,
            # so only fresh empty tiles are treated as reliable recharge targets.
            if self.current_tile_type(position) != TileType.EMPTY:
                continue
            if position not in distances:
                continue
            distance = distances[position]
            travel_energy = distance * self.config.unit_move_cost
            utility = self.known_energy(position) - travel_energy
            if utility <= 0:
                continue
            if self.candidate_mask[position]:
                utility += 1.0
            if utility > best_utility:
                best_utility = utility
                best = position
        return best

    def _reachable_distances(
        self, origin: Position, *, available_energy: int | None
    ) -> dict[Position, int]:
        distances = {origin: 0}
        reserves = {
            origin: float("inf")
            if available_energy is None
            else float(max(0, available_energy))
        }
        frontier: deque[Position] = deque([origin])
        while frontier:
            current = frontier.popleft()
            distance = distances[current]
            x, y = current
            for neighbor in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
                if not self.is_passable(neighbor):
                    continue
                reserve = reserves[current]
                if reserve != float("inf"):
                    if reserve < self.config.unit_move_cost:
                        continue
                    reserve -= self.config.unit_move_cost
                    field = min(0.0, self.known_energy(neighbor))
                    if self.current_tile_type(neighbor) == TileType.NEBULA:
                        # The reduction is hidden and may exceed the remaining
                        # reserve, so a low-energy recharge route cannot rely on
                        # traversing nebula safely.
                        continue
                    reserve += field
                    if reserve < 0:
                        continue
                new_distance = distance + 1
                old_distance = distances.get(neighbor)
                old_reserve = reserves.get(neighbor, float("-inf"))
                if old_distance is not None and (
                    old_distance < new_distance
                    or (old_distance == new_distance and old_reserve >= reserve)
                ):
                    continue
                distances[neighbor] = new_distance
                reserves[neighbor] = reserve
                frontier.append(neighbor)
        return distances

    def frontier_target(
        self,
        origin: Position,
        reserved: Iterable[Position] = (),
    ) -> Position:
        """Select a tile that exposes the most unknown or stale map information."""

        reserved_set = set(reserved)
        sensor = max(1, self.config.unit_sensor_range)
        own_corner = (
            (0, 0)
            if self.team_id == 0
            else (self.config.map_width - 1, self.config.map_height - 1)
        )
        best = origin
        best_score = float("-inf")

        for x in range(self.config.map_width):
            for y in range(self.config.map_height):
                position = x, y
                if not self.is_passable(position) or position in reserved_set:
                    continue
                x0, x1 = max(0, x - sensor), min(self.config.map_width, x + sensor + 1)
                y0, y1 = max(0, y - sensor), min(self.config.map_height, y + sensor + 1)
                seen = self.last_seen[x0:x1, y0:y1]
                unknown = int(np.count_nonzero(seen < 0))
                if unknown:
                    information = 2.5 * unknown
                else:
                    ages = np.maximum(0, self.step - seen)
                    information = 0.025 * float(np.sum(ages))
                distance = manhattan(origin, position)
                advance = manhattan(own_corner, position)
                score = (
                    information
                    - 0.28 * distance
                    - 0.45 * float(self.visits[position])
                    + 0.025 * advance
                )
                if score > best_score:
                    best_score = score
                    best = position
        return best
