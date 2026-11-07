# Route: the second intermediate object, one per pair per step, and the candidate search. A route is a chain
# of straight segments through free tiles, consecutive segments sharing their corner tile, so a straight route
# has one segment, an L two, and a detour three. The qubit tiles are never part of the route: the first tile
# is a vertical neighbour of the Z party and the last a horizontal neighbour of the X party, which is a qubit
# or the consumed magic tile, and a magic tile attaches from any side.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from . import constant
from .circuit import Pair
from .mapping import Coord, Mapping


class RouteError(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    start: Coord
    end: Coord

    def __post_init__(self):
        object.__setattr__(self, "start", tuple(self.start))
        object.__setattr__(self, "end", tuple(self.end))
        if self.start[0] != self.end[0] and self.start[1] != self.end[1]:
            raise RouteError(f"segment {self.start}->{self.end} is not straight")

    @property
    def length(self) -> int:
        return abs(self.end[0] - self.start[0]) + abs(self.end[1] - self.start[1]) + 1

    def tiles(self) -> list[Coord]:
        (row0, col0), (row1, col1) = self.start, self.end
        if row0 == row1:
            direction = 1 if col1 >= col0 else -1
            return [(row0, col) for col in range(col0, col1 + direction, direction)]
        direction = 1 if row1 >= row0 else -1
        return [(row, col0) for row in range(row0, row1 + direction, direction)]


@dataclass(frozen=True)
class Route:
    pair: int
    segments: tuple[Segment, ...]
    magic: Optional[Coord] = None   # the magic tile consumed, for (q, MAGIC) pairs

    def __post_init__(self):
        object.__setattr__(self, "segments", tuple(self.segments))
        if self.magic is not None:
            object.__setattr__(self, "magic", tuple(self.magic))
        if not self.segments:
            raise RouteError(f"route for pair {self.pair} has no segments")
        for segment, following in zip(self.segments, self.segments[1:]):
            if segment.end != following.start:
                raise RouteError(f"route for pair {self.pair}: segment ends at {segment.end}, next starts at {following.start}")

    @property
    def start(self) -> Coord:
        return self.segments[0].start

    @property
    def end(self) -> Coord:
        return self.segments[-1].end

    def tiles(self) -> list[Coord]:
        # Tiles in walking order, the shared corner listed once.
        out = list(self.segments[0].tiles())
        for segment in self.segments[1:]:
            out.extend(segment.tiles()[1:])
        return out

    @property
    def length(self) -> int:
        return len(self.tiles())


def check_route(route: Route, pair: Pair, mapping: Mapping) -> None:
    tiles = route.tiles()
    if len(set(tiles)) != len(tiles):
        raise RouteError(f"pair {route.pair}: route revisits a tile")
    for tile in tiles:
        if not mapping.is_free(tile):
            raise RouteError(f"pair {route.pair}: tile {tile} is not a free tile")
    if route.start not in mapping.z_side(pair.z):
        raise RouteError(f"pair {route.pair}: start {route.start} is not on the Z side of q{pair.z} at {mapping.qubits[pair.z]}")

    if pair.is_magic:
        if route.magic is None:
            raise RouteError(f"pair {route.pair}: T route names no magic tile")
        if route.magic not in mapping.magic:
            raise RouteError(f"pair {route.pair}: {route.magic} is not a magic tile")
        if route.end not in mapping.magic_side(route.magic):
            raise RouteError(f"pair {route.pair}: end {route.end} is not adjacent to magic {route.magic}")
    else:
        if route.magic is not None:
            raise RouteError(f"pair {route.pair}: CX route consumes a magic tile")
        if route.end not in mapping.x_side(pair.x):
            raise RouteError(f"pair {route.pair}: end {route.end} is not on the X side of q{pair.x} at {mapping.qubits[pair.x]}")


def segments_of(tiles: Sequence[Coord]) -> tuple[Segment, ...]:
    # One segment per maximal straight run of the path.
    if not tiles:
        raise RouteError("empty path")
    segments = []
    start = tiles[0]
    for previous, current, following in zip(tiles, tiles[1:], tiles[2:]):
        if (current[0] - previous[0], current[1] - previous[1]) != (following[0] - current[0], following[1] - current[1]):
            segments.append(Segment(start, current))
            start = current
    segments.append(Segment(start, tiles[-1]))
    return tuple(segments)


def l_routes(start: Coord, end: Coord) -> list[tuple[Segment, ...]]:
    # The straight chain, or the row-first and the column-first L.
    (row0, col0), (row1, col1) = start, end
    if row0 == row1 or col0 == col1:
        return [(Segment(start, end),)]
    return [(Segment(start, (row0, col1)), Segment((row0, col1), end)),
            (Segment(start, (row1, col0)), Segment((row1, col0), end))]


Schedule = list[list[Route]]


def free_routes(pair_id: int, starts: Sequence[Coord], ends: Sequence[Coord], mapping: Mapping,
                magic: Optional[Coord] = None, blocked: Sequence[Coord] = (),
                connectors: int = constant.ROUTE_CONNECTORS) -> list[Route]:
    # Three-segment routes are tried only for a start and end that no straight or L route joins, and the
    # connectors nearest lines that give a free route are kept, lines inside the span between start and end first.
    blocked = set(blocked)
    out = []

    def free(tiles):
        return all(mapping.is_free(tile) and tile not in blocked for tile in tiles)

    def segment_tiles(segments):
        return [tile for segment in segments for tile in segment.tiles()]

    for start in starts:
        if not mapping.is_free(start) or start in blocked:
            continue
        for end in ends:
            if not mapping.is_free(end) or end in blocked:
                continue
            direct = 0
            for segments in l_routes(start, end):
                if free(segment_tiles(segments)):
                    out.append(Route(pair_id, segments, magic))
                    direct += 1
            if direct:
                continue

            # Row, column, row through a connecting column, which on a shared line is the U-turn.
            low, high = sorted((start[1], end[1]))
            found = 0
            for col in sorted(range(mapping.width),
                              key=lambda col: (0 if low < col < high else min(abs(col - low), abs(col - high)), col)):
                if col in (start[1], end[1]):
                    continue
                segments = (Segment(start, (start[0], col)), Segment((start[0], col), (end[0], col)), Segment((end[0], col), end))
                if free(segment_tiles(segments)):
                    out.append(Route(pair_id, segments, magic))
                    found += 1
                    if found == connectors:
                        break

            # Column, row, column through a connecting row.
            low, high = sorted((start[0], end[0]))
            found = 0
            for row in sorted(range(mapping.height),
                              key=lambda row: (0 if low < row < high else min(abs(row - low), abs(row - high)), row)):
                if row in (start[0], end[0]):
                    continue
                segments = (Segment(start, (row, start[1])), Segment((row, start[1]), (row, end[1])), Segment((row, end[1]), end))
                if free(segment_tiles(segments)):
                    out.append(Route(pair_id, segments, magic))
                    found += 1
                    if found == connectors:
                        break
    return out


def shortest_route(pair_id: int, z_qubit: int, x_qubit: int, mapping: Mapping, blocked: Sequence[Coord] = ()) -> Optional[Route]:
    # Ties keep enumeration order.
    best = None
    for candidate in free_routes(pair_id, mapping.z_side(z_qubit), mapping.x_side(x_qubit), mapping, blocked=blocked):
        if best is None or candidate.length < best.length:
            best = candidate
    return best


def shortest_magic_route(pair_id: int, z_qubit: int, mapping: Mapping, blocked: Sequence[Coord] = (),
                         consumed: Sequence[Coord] = ()) -> Optional[Route]:
    # Magic tiles not in consumed are tried by distance, and the search stops once no closer tile can beat the best.
    z_tile = mapping.qubits[z_qubit]

    def distance(magic):
        return abs(magic[0] - z_tile[0]) + abs(magic[1] - z_tile[1])

    consumed = set(consumed)
    best = None
    for magic in sorted(mapping.magic, key=lambda magic: (distance(magic), magic)):
        if best is not None and distance(magic) - 1 >= best.length:
            break
        if magic in consumed:
            continue
        for candidate in free_routes(pair_id, mapping.z_side(z_qubit), mapping.magic_side(magic), mapping, magic=magic, blocked=blocked):
            if best is None or candidate.length < best.length:
                best = candidate
    return best
