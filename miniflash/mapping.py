# Mapping: the grid, where each qubit sits, and where the magic tiles are. Coordinates are (row, col) with
# row 0 at the top. Every qubit faces the same way, its Z boundary the top and bottom side and its X boundary
# the left and right side. A Mapping is the first intermediate object a solver produces, and together with
# the routes it derives the Program.
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Optional

Coord = tuple[int, int]


def _paired_positions(n_sites: int) -> tuple[list[int], int]:
    # Site coordinates 1, 2, 4, 5, 7, 8 and so on, with the extent depending on the parity of n_sites.
    positions = []
    position = 1
    for i in range(n_sites):
        positions.append(position)
        position += 1 if i % 2 == 0 else 2
    return positions, position


@dataclass(frozen=True)
class Mapping:
    height: int
    width: int
    qubits: dict[int, Coord]
    magic: tuple[Coord, ...] = ()
    _occupied: frozenset[Coord] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "qubits", dict(self.qubits))
        object.__setattr__(self, "magic", tuple(tuple(tile) for tile in self.magic))
        object.__setattr__(self, "_occupied", frozenset(list(self.qubits.values()) + list(self.magic)))

    def in_bounds(self, tile: Coord) -> bool:
        return 0 <= tile[0] < self.height and 0 <= tile[1] < self.width

    def is_free(self, tile: Coord) -> bool:
        return self.in_bounds(tile) and tile not in self._occupied

    @property
    def occupied(self) -> frozenset[Coord]:
        return self._occupied

    def vertical(self, tile: Coord) -> tuple[Coord, ...]:
        row, col = tile
        return tuple(neighbour for neighbour in ((row - 1, col), (row + 1, col)) if self.in_bounds(neighbour))

    def horizontal(self, tile: Coord) -> tuple[Coord, ...]:
        row, col = tile
        return tuple(neighbour for neighbour in ((row, col - 1), (row, col + 1)) if self.in_bounds(neighbour))

    def z_side(self, qubit: int) -> tuple[Coord, ...]:
        return self.vertical(self.qubits[qubit])

    def x_side(self, qubit: int) -> tuple[Coord, ...]:
        return self.horizontal(self.qubits[qubit])

    def magic_side(self, magic: Coord) -> tuple[Coord, ...]:
        # A magic state is prepared in whatever orientation suits, so any neighbour attaches.
        return self.vertical(magic) + self.horizontal(magic)

    @classmethod
    def sparse(cls, site_rows: int, site_cols: int, n_qubits: int, order: Iterable[int] = None) -> Mapping:
        # Sites at even coordinates from 2 with one free tile between neighbours, a padding ring, and a magic tile
        # on every other ring tile; sites fill row-major in order and the trailing sites stay empty.
        height = 2 * site_rows + 3
        width = 2 * site_cols + 3
        sites = [(row, col) for row in range(2, height - 2, 2) for col in range(2, width - 2, 2)]
        order = list(order) if order is not None else list(range(n_qubits))
        qubits = {qubit: sites[i] for i, qubit in enumerate(order)}
        tiles = ([(0, col) for col in range(1, width, 2)] + [(row, width - 1) for row in range(1, height, 2)]
                 + [(height - 1, col) for col in range(1, width, 2)] + [(row, 0) for row in range(1, height, 2)])
        return cls(height, width, qubits, tuple(dict.fromkeys(tiles)))

    @classmethod
    def paired(cls, site_rows: int, site_cols: int, n_qubits: int, order: Iterable[int] = None) -> Mapping:
        # Every two site rows share one lane row and every two site columns one lane column, with a lane before
        # the first site, no padding ring, and a magic tile at both ends of every site row.
        rows, height = _paired_positions(site_rows)
        cols, inner_width = _paired_positions(site_cols)
        if site_cols % 2 == 1:
            inner_width += 1
        cols = [col + 1 for col in cols]
        width = inner_width + 2

        sites = [(row, col) for row in rows for col in cols]
        order = list(order) if order is not None else list(range(n_qubits))
        qubits = {qubit: sites[i] for i, qubit in enumerate(order)}
        return cls(height, width, qubits, tuple((row, col) for row in rows for col in (0, width - 1)))


# The virtual map behind a Mapping: it fixes the qubit order as virtual sites (i, j) on a site array and
# nothing else, while coordinates come from the array geometry and from compaction, which keeps this order.
@dataclass(frozen=True)
class Layout:
    site_rows: int
    site_cols: int
    order: dict[int, tuple[int, int]]   # qubit to its virtual site (i, j)

    def __post_init__(self):
        object.__setattr__(self, "order", {qubit: tuple(site) for qubit, site in self.order.items()})

    @classmethod
    def identity(cls, n_qubits: int, shape: Optional[tuple[int, int]] = None) -> Layout:
        # Qubit q at site (q // cols, q % cols) on a rows x cols site array, the smallest square by default.
        rows, cols = shape if shape is not None else (math.ceil(math.sqrt(n_qubits)),) * 2
        return cls(rows, cols, {qubit: (qubit // cols, qubit % cols) for qubit in range(n_qubits)})
