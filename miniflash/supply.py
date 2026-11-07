# Magic-state supply geometry: factories, delivery rings, and one path per consumed state. The chip is the
# program's H x W grid, and with K factories it is wrapped in K rings of tiles, ring i running along rows -i
# and H - 1 + i and columns -i and W - 1 + i, each ring belonging to one factory. Factories sit outside the
# outermost ring, placed clockwise from the top-left corner one tile apart, each with one port in the middle
# of its face towards the rings; a port near a corner lies beyond the ends of the inner rings, so rings are
# dealt out with the factories nearest the corners taking the outermost ones. A consumed state travels from
# its port straight in to its ring, around the ring the short way to the magic tile's row on the magic tile's
# side of the chip, then straight in along that row, and the k-th state consumed goes to factory k mod K.
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .factory import Factory
from .mapping import Coord
from .program import IDLE, MAGIC, Program


@dataclass(frozen=True)
class Delivery:
    step: int
    factory: int
    magic: Coord
    tiles: tuple[Coord, ...]   # port, ..., magic tile, in walking order


@dataclass(frozen=True)
class Box:
    side: str       # top, right, bottom or left
    row: int        # top-left tile
    col: int
    height: int
    width: int
    port: Coord     # middle tile of the face towards the rings
    ring: int = 0   # the ring this factory delivers on, 1 being the innermost


# conflicts records every tile two deliveries share within a step and every used chip tile a delivery crosses.
@dataclass
class Supply:
    rings: int
    factory: Factory
    boxes: list[Box] = field(default_factory=list)
    deliveries: list[Delivery] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


def ring_tiles(ring: int, height: int, width: int) -> list[Coord]:
    # The ring clockwise from its top-left corner.
    top = -ring
    bottom = height - 1 + ring
    left = -ring
    right = width - 1 + ring
    out = [(top, col) for col in range(left, right + 1)]
    out += [(row, right) for row in range(top + 1, bottom + 1)]
    out += [(bottom, col) for col in range(right - 1, left - 1, -1)]
    out += [(row, left) for row in range(bottom - 1, top, -1)]
    return out


def place_factories(count: int, height: int, width: int, factory: Factory) -> list[Box]:
    # A factory is factory.height tiles deep away from the chip and factory.width along its side.
    rings = count
    depth = factory.height
    span = factory.width
    boxes: list[Box] = []
    for side, length in (("top", width + 2 * rings), ("right", height + 2 * rings),
                         ("bottom", width + 2 * rings), ("left", height + 2 * rings)):
        offset = 0
        while len(boxes) < count and offset + span <= length:
            if side == "top":
                row, col = -rings - depth, -rings + offset
                boxes.append(Box(side, row, col, depth, span, (row + depth - 1, col + span // 2)))
            elif side == "right":
                row, col = -rings + offset, width + rings
                boxes.append(Box(side, row, col, span, depth, (row + span // 2, col)))
            elif side == "bottom":
                row, col = height + rings, width + rings - offset - span
                boxes.append(Box(side, row, col, depth, span, (row, col + span // 2)))
            else:
                row, col = height + rings - offset - span, -rings - depth
                boxes.append(Box(side, row, col, span, depth, (row + span // 2, col + depth - 1)))
            offset += span + 1
    return assign_rings(boxes, height, width)


def innermost_ring(box: Box, height: int, width: int) -> int:
    # The innermost ring the port's straight line in meets.
    row, col = box.port
    if box.side in ("top", "bottom"):
        return max(1, -col, col - (width - 1))
    return max(1, -row, row - (height - 1))


def assign_rings(boxes: list[Box], height: int, width: int) -> list[Box]:
    # Rings K down to 1 go to the factories whose lines meet the fewest rings first.
    n_factories = len(boxes)
    order = sorted(range(n_factories),
                   key=lambda factory_index: (-innermost_ring(boxes[factory_index], height, width), factory_index))
    out = list(boxes)
    for ring, factory_index in zip(range(n_factories, 0, -1), order):
        out[factory_index] = replace(boxes[factory_index], ring=ring)
    return out


def _inward(box: Box, ring: int, height: int, width: int) -> list[Coord]:
    # From the port straight in to the ring, crossing the rings outside it.
    row, col = box.port
    if box.side == "top":
        return [(path_row, col) for path_row in range(row + 1, -ring + 1)]
    if box.side == "bottom":
        return [(path_row, col) for path_row in range(row - 1, height - 2 + ring, -1)]
    if box.side == "left":
        return [(row, path_col) for path_col in range(col + 1, -ring + 1)]
    return [(row, path_col) for path_col in range(col - 1, width - 2 + ring, -1)]


def _around(ring: list[Coord], start: Coord, end: Coord) -> list[Coord]:
    # Tiles of the ring from start to end the short way, both included.
    start_index = ring.index(start)
    end_index = ring.index(end)
    n_tiles = len(ring)
    forward = [ring[(start_index + offset) % n_tiles] for offset in range((end_index - start_index) % n_tiles + 1)]
    backward = [ring[(start_index - offset) % n_tiles] for offset in range((start_index - end_index) % n_tiles + 1)]
    return forward if len(forward) <= len(backward) else backward


def delivery_path(box: Box, ring: int, magic: Coord, height: int, width: int) -> list[Coord]:
    # Port, in to the ring, around it, then along the magic tile's row to the magic tile.
    row, col = magic
    left = col < width / 2
    exit_tile = (row, -ring) if left else (row, width - 1 + ring)
    radial = _inward(box, ring, height, width)
    tiles = ring_tiles(ring, height, width)
    around = _around(tiles, radial[-1], exit_tile)
    if left:
        inward = [(row, path_col) for path_col in range(-ring + 1, col)]
    else:
        inward = [(row, path_col) for path_col in range(width - 2 + ring, col, -1)]
    return [box.port] + radial[:-1] + around + inward + [magic]


def plan(program: Program, factories: int, factory: Factory = Factory()) -> Supply:
    height = program.height
    width = program.width
    supply = Supply(factories, factory, place_factories(factories, height, width, factory))
    n_consumed = 0
    for step_index, step in enumerate(program.steps):
        used = {}
        for vertex in sorted(step.of_kind(MAGIC), key=lambda vertex: vertex.tile):
            if vertex.id == IDLE:
                continue
            factory_index = n_consumed % factories
            n_consumed += 1
            box = supply.boxes[factory_index]
            path = delivery_path(box, box.ring, vertex.tile, height, width)
            supply.deliveries.append(Delivery(step_index, factory_index, vertex.tile, tuple(path)))

            for tile in path:
                if tile in used:
                    supply.conflicts.append(f"step {step_index}: factories {used[tile]} and {factory_index} both use {tile}")
                used[tile] = factory_index
                inside = 0 <= tile[0] < height and 0 <= tile[1] < width
                if inside and tile != vertex.tile and tile in step.vertices:
                    supply.conflicts.append(f"step {step_index}: factory {factory_index} crosses {tile}, used by the chip")
    return supply
