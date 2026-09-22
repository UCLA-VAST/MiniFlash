"""Cell synthesis.

Turns one region into a lattice-surgery cell via LaSsynth: stim extracts
the region's Choi dual stabilizers, boundary Paulis are solved over
GF(2), and a SAT solver (kissat when available, z3 otherwise) searches a
depth-minimal geometry that realizes them **at the exact ports the
floorplan assigned**.

Solutions land in a disk cache split into regimes. Every cell's SAT
problem is described by one :class:`CellSpec`, which is the single
source of both the solver spec and the cache key (sha256 of the
canonical dict, sorted keys, default fields omitted), so a constraint
cannot change without moving the key. ``canon.v1/`` holds fixed-columns
cells solved in canonical form — local qubits ordered by home column,
compact ports, wire-row crossings forbidden between all columns — so
the key degenerates to (qubit count, column-basis unsigned dual
stabilizers) and entries are independent of the permutation and of the
spectator gaps they are later stretched over. ``free.v1/`` holds cells
with floorplan-pinned ports, keyed by the full spec. ``perms.v1/``
holds fixed-columns permutations. Gate-order noise, Pauli frames and the wire ledger's pending
Hadamards share one cell in every regime: a wire arriving with its bit
set is brought back to standard colour by the channel's flip landing —
the wall that applies the owed Hadamard — so every cell body is solved
for the region's own gates on standard-coloured ports, and a cell's
``in_bases`` only records what the ledger delivered.

Budgets are enforced with a hard ``SIGALRM`` wall. Exhaustion raises
:class:`SynthTimeout` (the coarse portfolio reacts by splitting the
region), infeasibility raises :class:`SynthUnsat`; a failed stabilizer
verification retries at greater depth.
"""
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import stim
from lassynth import LatticeSurgerySynthesizer

from dataclasses import dataclass

from .floorplan import solve_floorplan, passthrough_floorplan
from .schedule import schedule_layers

STRIDE = 2
AXES = ("I", "J", "K")
PARITY_NONE = -1
AXIS_INDEX = {"I": 0, "J": 1, "K": 2}


@dataclass
class Cell:
    """A synthesized lattice-surgery cell: geometry blocks plus port pins."""

    #: [I, J, K] extents in cube units
    dims: list
    #: port pin dicts — name ("in_k"/"out_k"), offset, dir, parity
    pins: list
    #: geometry blocks (pipes, hadamards) in cell-local coordinates
    blocks: list
    #: Pauli correction carried by this instance (recomputed per use)
    pauli_frame: str = "I"
    #: {qubit: 0|1} wire ledger bits this instance was placed against (recomputed per use; None = unknown)
    in_bases: dict = None
    #: magic-state ports in circuit order ({qubit, dagger, basis, offset}), None = solver cell
    injections: list = None

    @classmethod
    def from_dict(cls, data):
        """Rebuild a Cell from a to_dict() payload.

        :param data: dict from :meth:`to_dict`.
        :returns: Cell.
        """
        return cls(dims=data["dims"], pins=data["pins"], blocks=data["blocks"], pauli_frame=data.get("pauli_frame", "I"), injections=data.get("injections"))

    def to_dict(self):
        """Serialize the Cell geometry to a JSON-safe dict (pauli_frame excluded — it is recomputed per use)."""
        payload = {"dims": self.dims, "pins": self.pins, "blocks": self.blocks}
        if self.injections is not None:
            payload["injections"] = self.injections
        return payload

    def pin_parities(self, columns, name_prefix):
        """Read the wire parity bit each qubit gets at this cell's pins.

        :param columns: {qubit: local column}.
        :param name_prefix: "in" | "out".
        :returns: {qubit: 0|1} (0 for qubits with no matching pin).
        """
        parity_by_offset = {}
        for pin in self.pins:
            if pin["name"].startswith(name_prefix):
                parity_by_offset[pin["offset"][0]] = max(0, int(pin.get("parity", 0)))
        return {qubit: parity_by_offset.get(column * STRIDE, 0) for qubit, column in columns.items()}


def _array_dimensions(array):
    return len(array), len(array[0]), len(array[0][0])


def _grid_position(i, j, k):
    return i * STRIDE, j * STRIDE, k * STRIDE


def _pipe_position(i, j, k, axis_index):
    position = [i * STRIDE, j * STRIDE, k * STRIDE]
    position[axis_index] += 1
    return tuple(position)


def _color_letter(color_value):
    return "r" if color_value == 0 else "b"


def _color_flip(color_value):
    return "b" if color_value == 0 else "r"


def _pipe_colors(axis, color_value):
    base = _color_letter(color_value)
    flipped = _color_flip(color_value)
    if axis == "I":
        return "_" + flipped + base
    if axis == "J":
        return base + "_" + flipped
    return flipped + base + "_"


def _block_faces(axis, colors):
    color_i, color_j, color_k = colors[0], colors[1], colors[2]
    if axis == "I":
        return "__" + color_j + color_j + color_k + color_k
    if axis == "J":
        return color_i + color_i + "__" + color_k + color_k
    return color_i + color_i + color_j + color_j + "__"


def _face_flip(color):
    if color == "r":
        return "b"
    if color == "b":
        return "r"
    raise RuntimeError(f"_face_flip: uncolored face {color!r}")


def _hadamard_faces_from_cube(axis, cube_faces):
    if axis != "K":
        raise RuntimeError(f"_hadamard_faces_from_cube: non-K hadamard segment (axis {axis!r}); only ColorKP/ColorKM can differ")
    pairs = ((0, 1), (2, 3))
    colors = [None, None]
    for pair_index, (first_face, second_face) in enumerate(pairs):
        for face_index in (first_face, second_face):
            color = cube_faces[face_index]
            if color in ("r", "b"):
                colors[pair_index] = color
                break
    if colors[0] is None and colors[1] is not None:
        colors[0] = _face_flip(colors[1])
    elif colors[1] is None and colors[0] is not None:
        colors[1] = _face_flip(colors[0])
    elif colors[0] is None and colors[1] is None:
        return "______", "______"

    minus_faces, plus_faces = ["_"] * 6, ["_"] * 6
    for pair_index, (first_face, second_face) in enumerate(pairs):
        color = colors[pair_index]
        flipped = _face_flip(color)
        minus_faces[first_face] = minus_faces[second_face] = color
        plus_faces[first_face] = plus_faces[second_face] = flipped

    return "".join(minus_faces), "".join(plus_faces)


def _scan_segments(lasre, axis):
    exists = lasre["Exist" + axis]
    size_i, size_j, size_k = _array_dimensions(exists)
    if axis == "K":
        color_plus = lasre.get("ColorKP")
        color_minus = lasre.get("ColorKM")
    else:
        color_plus = lasre.get("Color" + axis)
        color_minus = color_plus
    has_color_plus = bool(color_plus)
    has_color_minus = bool(color_minus)
    run_length = (size_i, size_j, size_k)[AXIS_INDEX[axis]]

    def access(position, a, b):
        if axis == "I":
            return position, a, b
        if axis == "J":
            return a, position, b
        return a, b, position

    if axis == "I":
        outer = [(j, k) for j in range(size_j) for k in range(size_k)]
    elif axis == "J":
        outer = [(i, k) for i in range(size_i) for k in range(size_k)]
    else:
        outer = [(i, j) for i in range(size_i) for j in range(size_j)]

    segments = []
    for a, b in outer:
        def make_segment(start, end, plus, minus, a=a, b=b):
            segment = {"axis": axis, "ijk": list(access(start, a, b)), "length": end - start, "colors": _pipe_colors(axis, plus)}
            if plus != minus:
                segment["hadamard"] = True
                segment["colors_m"] = _pipe_colors(axis, minus)
            return segment

        start, plus, minus = -1, -1, -1
        for position in range(run_length):
            i, j, k = access(position, a, b)
            exists_here = exists[i][j][k]
            current_plus = (color_plus[i][j][k] if has_color_plus else 0) if exists_here else -1
            current_minus = (color_minus[i][j][k] if has_color_minus else 0) if exists_here else -1
            if exists_here and start >= 0 and current_plus == plus and current_minus == minus:
                continue
            if start >= 0:
                segments.append(make_segment(start, position, plus, minus))
            if exists_here:
                start, plus, minus = position, current_plus, current_minus
            else:
                start, plus, minus = -1, -1, -1
        if start >= 0:
            segments.append(make_segment(start, run_length, plus, minus))

    return segments


def _extract_y_nodes(lasre):
    node_y = lasre["NodeY"]
    size_i, size_j, size_k = _array_dimensions(node_y)
    return [(i, j, k) for i in range(size_i) for j in range(size_j) for k in range(size_k) if node_y[i][j][k]]


def _assert_no_dead_ends(lasre):
    size_i, size_j, size_k = lasre["n_i"], lasre["n_j"], lasre["n_k"]
    protected = {(port["i"], port["j"], port["k"]) for port in lasre["ports"]}
    for port in lasre["ports"]:
        external = [port["i"], port["j"], port["k"]]
        external[AXIS_INDEX[port["d"]]] += 1 if port["e"] == "+" else -1
        protected.add(tuple(external))
    protected.update(_extract_y_nodes(lasre))

    exist_i = lasre.get("ExistI")
    exist_j = lasre.get("ExistJ")
    exist_k = lasre.get("ExistK")
    for i in range(size_i):
        for j in range(size_j):
            for k in range(size_k):
                if (i, j, k) in protected:
                    continue
                degree = 0
                if exist_i and i > 0 and exist_i[i - 1][j][k]:
                    degree += 1
                if exist_i and i < size_i - 1 and exist_i[i][j][k]:
                    degree += 1
                if exist_j and j > 0 and exist_j[i][j - 1][k]:
                    degree += 1
                if exist_j and j < size_j - 1 and exist_j[i][j][k]:
                    degree += 1
                if exist_k and k > 0 and exist_k[i][j][k - 1]:
                    degree += 1
                if exist_k and k < size_k - 1 and exist_k[i][j][k]:
                    degree += 1
                if degree == 1:
                    raise RuntimeError(f"_assert_no_dead_ends: dead end at {(i, j, k)}")


def _convert_ports(lasre):
    ports = []
    for port in lasre["ports"]:
        ports.append({"ijk": [port["i"] * STRIDE, port["j"] * STRIDE, port["k"] * STRIDE], "d": port["d"], "e": port["e"], "c": port["c"], "f": port["f"]})
    return ports


def _convert_blocks(lasre):
    blocks = []
    cube_index_by_position = {}
    hadamard_segments = []
    pin_external_positions = set()
    for port in lasre["ports"]:
        position = [port["i"], port["j"], port["k"]]
        position[AXIS_INDEX[port["d"]]] += 1 if port["e"] == "+" else -1
        pin_external_positions.add(tuple(position))

    def merge_cube(position_key, faces_string):
        if position_key in pin_external_positions:
            return
        block_index = cube_index_by_position.get(position_key)
        if block_index is not None:
            existing_faces = list(blocks[block_index]["faces"])
            for face_index in range(6):
                face_color = faces_string[face_index]
                if face_color not in ("_", "g"):
                    existing_faces[face_index] = face_color
            blocks[block_index]["faces"] = "".join(existing_faces)
        else:
            merged_faces = ["_"] * 6
            for face_index in range(6):
                if faces_string[face_index] != "_":
                    merged_faces[face_index] = faces_string[face_index]
            cube_index_by_position[position_key] = len(blocks)
            blocks.append({"pos": list(_grid_position(*position_key)), "faces": "".join(merged_faces), "block_type": "cube", "axis": None})


    pipe_endpoints = set()
    for axis in AXES:
        axis_index = AXIS_INDEX[axis]
        for segment in _scan_segments(lasre, axis):
            if segment.get("hadamard"):
                hadamard_segments.append((axis, segment))
                continue
            ijk = segment["ijk"]
            length = segment["length"]
            faces = _block_faces(axis, segment["colors"])
            for step in range(length):
                position = list(ijk)
                position[axis_index] += step
                blocks.append({"pos": list(_pipe_position(position[0], position[1], position[2], axis_index)), "faces": faces, "block_type": "pipe", "axis": axis})
                start_key = tuple(position)
                pipe_endpoints.add(start_key)
                merge_cube(start_key, faces)
                if step == length - 1:
                    end_position = list(position)
                    end_position[axis_index] += 1
                    end_key = tuple(end_position)
                    pipe_endpoints.add(end_key)
                    merge_cube(end_key, faces)

    for y_node in _extract_y_nodes(lasre):
        block_index = cube_index_by_position.get(y_node)
        if block_index is not None:
            blocks[block_index]["block_type"] = "ynode"
        elif y_node in pipe_endpoints:
            cube_index_by_position[y_node] = len(blocks)
            blocks.append({"pos": list(_grid_position(*y_node)), "faces": "______", "block_type": "ynode", "axis": None})

    for axis, segment in hadamard_segments:
        axis_index = AXIS_INDEX[axis]
        ijk = segment["ijk"]
        length = segment["length"]
        colors = segment["colors"]
        colors_minus = segment.get("colors_m") or colors
        for step in range(length):
            position = list(ijk)
            position[axis_index] += step
            start_key = tuple(position)
            end_position = list(position)
            end_position[axis_index] += 1
            end_key = tuple(end_position)
            start_cube_faces = "______"
            start_cube_index = cube_index_by_position.get(start_key)
            if start_cube_index is not None:
                start_cube_faces = blocks[start_cube_index]["faces"]
            minus_faces, plus_faces = _hadamard_faces_from_cube(axis, start_cube_faces)
            if minus_faces == "______" and plus_faces == "______":
                minus_faces = _block_faces(axis, colors_minus)
                plus_faces = _block_faces(axis, colors)
            blocks.append({"pos": list(_pipe_position(position[0], position[1], position[2], axis_index)), "faces": minus_faces, "block_type": "hadamard", "axis": axis})
            pipe_endpoints.add(start_key)
            pipe_endpoints.add(end_key)
            merge_cube(start_key, minus_faces)
            merge_cube(end_key, plus_faces)

    occupied = {tuple(_grid_position(*key)) for key in cube_index_by_position}
    for block in blocks:
        if block["block_type"] in ("pipe", "hadamard"):
            occupied.add(tuple(block["pos"]))

    port_exit_faces = set()
    for port in lasre["ports"]:
        position_key = (port["i"], port["j"], port["k"])
        axis_index = AXIS_INDEX[port["d"]]
        face_index = 2 * axis_index + (1 if port["e"] == "+" else 0)
        port_exit_faces.add((position_key, axis_index, face_index))

    def cap_block(block_index, position_key):
        block = blocks[block_index]
        faces = list(block["faces"])
        position = block["pos"]
        changed = False
        pipe_axis_index = AXIS_INDEX[block["axis"]] if block["axis"] else -1
        for axis_index in range(3):
            face_minus, face_plus = 2 * axis_index, 2 * axis_index + 1
            if block["block_type"] != "cube" and axis_index == pipe_axis_index:
                continue
            if faces[face_minus] != "_" or faces[face_plus] != "_":
                continue
            neighbor_minus = list(position)
            neighbor_minus[axis_index] -= 1
            neighbor_plus = list(position)
            neighbor_plus[axis_index] += 1
            has_minus = tuple(neighbor_minus) in occupied
            has_plus = tuple(neighbor_plus) in occupied
            if has_minus and has_plus:
                continue
            cap_color = None
            for other_axis in range(3):
                if other_axis == axis_index:
                    continue
                color = faces[2 * other_axis]
                if color in ("r", "b"):
                    cap_color = color
                    break
            if cap_color is None:
                continue
            if not has_minus and not (position_key is not None and (position_key, axis_index, face_minus) in port_exit_faces):
                faces[face_minus] = cap_color
            if not has_plus and not (position_key is not None and (position_key, axis_index, face_plus) in port_exit_faces):
                faces[face_plus] = cap_color
            changed = True
        if changed:
            block["faces"] = "".join(faces)

    for position_key, block_index in list(cube_index_by_position.items()):
        cap_block(block_index, position_key)
    for block_index, block in enumerate(blocks):
        if block["block_type"] == "pipe":
            cap_block(block_index, None)

    existing_pipe_positions = {tuple(block["pos"]) for block in blocks if block["block_type"] in ("pipe", "hadamard")}
    for port in lasre["ports"]:
        port_position = [port["i"], port["j"], port["k"]]
        axis_index = AXIS_INDEX[port["d"]]
        external = list(port_position)
        external[axis_index] += 1 if port["e"] == "+" else -1
        pipe_lasre_position = [min(port_position[coordinate], external[coordinate]) for coordinate in range(3)]
        pipe_geometry_position = _pipe_position(pipe_lasre_position[0], pipe_lasre_position[1], pipe_lasre_position[2], axis_index)
        if pipe_geometry_position in existing_pipe_positions:
            continue
        port_key = tuple(port_position)
        cube_faces = "______"
        cube_block_index = cube_index_by_position.get(port_key)
        if cube_block_index is not None:
            cube_faces = blocks[cube_block_index]["faces"]
        pipe_faces = list(cube_faces)
        pipe_faces[2 * axis_index] = "_"
        pipe_faces[2 * axis_index + 1] = "_"
        blocks.append({"pos": list(pipe_geometry_position), "faces": "".join(pipe_faces), "block_type": "pipe", "axis": AXES[axis_index]})

    for port in lasre["ports"]:
        port_key = (port["i"], port["j"], port["k"])
        cube_block_index = cube_index_by_position.get(port_key)
        if cube_block_index is None:
            continue
        axis_index = AXIS_INDEX[port["d"]]
        face_index = 2 * axis_index + (1 if port["e"] == "+" else 0)
        faces = list(blocks[cube_block_index]["faces"])
        if faces[face_index] != "_":
            faces[face_index] = "_"
            blocks[cube_block_index]["faces"] = "".join(faces)

    return blocks


def _pins_from_ports(ports):
    pins = []
    seen_names = set()
    for port in ports:
        name = port["f"].replace(" ", "_")
        if name in seen_names:
            raise RuntimeError(f"_pins_from_ports: duplicate port name {name!r}")
        seen_names.add(name)
        ijk = port["ijk"]
        pins.append({"name": name, "dir": port["d"], "end": port["e"], "offset": [ijk[0], ijk[1], ijk[2]]})
    return pins


def _transverse_axis_first(axis):
    return "J" if axis == "I" else "I"


def _transverse_axis_second(axis):
    return "J" if axis == "K" else "K"


def _parity_from_face_colors(first_color, second_color):
    if first_color == "r":
        return 1
    if first_color == "b":
        return 0
    raise RuntimeError(f"_parity_from_face_colors: uncolored transverse faces {first_color!r}/{second_color!r}")


def _parity_from_faces(faces, axis):
    return _parity_from_face_colors(faces[2 * AXIS_INDEX[_transverse_axis_first(axis)]], faces[2 * AXIS_INDEX[_transverse_axis_second(axis)]])


def _parity_flip(parity):
    return parity if parity < 0 else 1 - parity


def _block_from_lasre_block(lasre_block):
    position = list(lasre_block["pos"])
    axis = lasre_block["axis"] or "I"
    if lasre_block["block_type"] == "pipe":
        return {"kind": "pipe", "pos": position, "axis": axis, "parity": _parity_from_faces(lasre_block["faces"], axis)}
    if lasre_block["block_type"] == "hadamard":
        return {"kind": "hadamard", "pos": position, "axis": axis, "parity": _parity_from_faces(lasre_block["faces"], axis)}
    if lasre_block["block_type"] == "ynode":
        return {"kind": "ynode", "pos": position, "parity": PARITY_NONE}
    raise RuntimeError(f"_block_from_lasre_block: unexpected block type {lasre_block['block_type']!r}")


def _pin_pipe_position(offset, direction, end):
    sign = 1 if end == "+" else -1
    position = list(offset)
    position[AXIS_INDEX[direction]] += sign
    return position


def _resolve_pins(pins, blocks, name):
    resolved = []
    pin_pipe_indices = []
    for pin in pins:
        pipe_position = _pin_pipe_position(pin["offset"], pin["dir"], pin["end"])
        parity = PARITY_NONE
        for block in blocks:
            if block["pos"] != pipe_position:
                continue
            if block["kind"] == "hadamard":
                parity = block["parity"] if pin["end"] == "+" else _parity_flip(block["parity"])
            else:
                parity = block["parity"]
            break
        if parity < 0:
            raise RuntimeError(f"_resolve_pins: no pin pipe at {pipe_position} for {name}.{pin['name']}")

        for block_index, block in enumerate(blocks):
            if block["pos"] != pipe_position:
                continue
            if block["kind"] in ("pipe", "hadamard"):
                pin_pipe_indices.append(block_index)
                continue
            raise RuntimeError(f"_resolve_pins: unexpected block at pin pipe coord: {name}.{pin['name']}")
        resolved.append({"name": pin["name"], "dir": pin["dir"], "parity": parity, "pipe_pos": pipe_position})

    for block_index in sorted(set(pin_pipe_indices), reverse=True):
        del blocks[block_index]
    return resolved


def _block_bounds(block):
    position = block["pos"]
    low = list(position)
    high = [position[0] + 1, position[1] + 1, position[2] + 1]
    if block["kind"] in ("pipe", "hadamard"):
        axis_index = AXIS_INDEX[block["axis"]]
        low[axis_index] -= 1
        high[axis_index] += 1
    return low, high


def _assert_coords_in_bounds(blocks, dims):
    for block in blocks:
        low, high = _block_bounds(block)
        for coordinate in range(3):
            if low[coordinate] < 0 or high[coordinate] > dims[coordinate]:
                raise RuntimeError(f"_assert_coords_in_bounds: block {block} outside dims {dims}")


def _map_columns(cell, column_of):
    width_tiles = (cell.dims[0] + 1) // 2
    blocks = []
    for block in cell.blocks:
        i = block["pos"][0]
        if block["kind"] in ("pipe", "hadamard") and block["axis"] == "I":
            left = (i - 1) // 2
            start, stop = 2 * column_of[left] + 1, 2 * column_of[left + 1]
            for position, center in enumerate(range(start, stop, 2)):
                segment = dict(block, pos=[center, block["pos"][1], block["pos"][2]])
                if position > 0 and block["kind"] == "hadamard":
                    segment["kind"] = "pipe"
                blocks.append(segment)
        else:
            blocks.append(dict(block, pos=[2 * column_of[i // 2] + i % 2, block["pos"][1], block["pos"][2]]))
    pins = [dict(pin, offset=[2 * column_of[pin["offset"][0] // 2] + pin["offset"][0] % 2, pin["offset"][1], pin["offset"][2]]) for pin in cell.pins]
    dims = [2 * (column_of[width_tiles - 1] + 1) - 1, cell.dims[1], cell.dims[2]]
    return Cell(dims=dims, pins=pins, blocks=blocks, pauli_frame=cell.pauli_frame, in_bases=cell.in_bases)


def from_lasre(lasre, name):
    """Convert a LaSsynth lasre solution dict into a Cell.

    :param lasre: dict, LaSsynth result (n_i/n_j/n_k grids, ports, blocks).
    :param name: str, used in error messages.
    :returns: Cell (dims, resolved pins with parity, pipe/hadamard blocks).
    :raises RuntimeError: on dead ends, out-of-bounds blocks or non-identity
        boundary fixup.
    """
    _assert_no_dead_ends(lasre)
    dims = [lasre["n_i"] * STRIDE - 1, lasre["n_j"] * STRIDE - 1, lasre["n_k"] * STRIDE - 1]
    lasre_blocks = _convert_blocks(lasre)
    lasre_pins = _pins_from_ports(_convert_ports(lasre))

    fixup = ""
    optional = lasre.get("optional")
    if isinstance(optional, dict):
        fixup = optional.get("boundary_pauli_fixup", "")
    for character in fixup:
        if character != "I":
            raise RuntimeError("from_lasre: non-identity boundary_pauli_fixup requires sfix pin extension, not supported (Clifford-only build)")

    blocks = [_block_from_lasre_block(lasre_block) for lasre_block in lasre_blocks if lasre_block["block_type"] != "cube"]
    resolved_pins = _resolve_pins(lasre_pins, blocks, name)
    _assert_coords_in_bounds(blocks, dims)

    pins = []
    if blocks:
        for pin in resolved_pins:
            axis_index = AXIS_INDEX[pin["dir"]]
            position_value = pin["pipe_pos"][axis_index]
            dimension_value = dims[axis_index]
            end = "+" if position_value >= dimension_value - 1 else "-"
            pins.append({"name": pin["name"], "dir": pin["dir"], "end": end, "parity": pin["parity"], "offset": pin["pipe_pos"]})

    return Cell(dims=dims, pins=pins, blocks=blocks)

Z3_TIMEOUT_MS = 60_000
KISSAT_DIR = os.environ.get("MINIFLASH_KISSAT_DIR") or os.path.dirname(shutil.which("kissat") or "")


class SynthTimeout(RuntimeError):
    pass


class SynthUnsat(RuntimeError):
    pass


def _default_solver():
    forced = os.environ.get("MINIFLASH_SOLVER")
    if forced:
        return forced
    for _ in range(3):
        if os.path.isfile(os.path.join(KISSAT_DIR, "kissat")):
            return "kissat"
        time.sleep(0.5)
    print("miniflash: kissat wrapper not visible, falling back to z3 (slow)", file=sys.stderr)
    return "z3"


def _make_synthesizer(solver):
    if solver == "kissat":
        return LatticeSurgerySynthesizer(solver="kissat", kissat_dir=KISSAT_DIR)
    import z3
    z3.set_param("timeout", Z3_TIMEOUT_MS)
    return LatticeSurgerySynthesizer(solver="z3")


def _local_gates(region, order=None):
    local_index = {qubit: index for index, qubit in enumerate(order if order is not None else region.qubits)}
    return [(name, tuple(local_index[qubit] for qubit in qubits)) for name, qubits in region.gates]


CACHE_CANON = "canon.v1"
CACHE_FREE = "free.v1"
CACHE_PERMS = "perms.v1"


@dataclass
class CellSpec:
    """Single source of truth for one cell's SAT problem.

    Both the solver spec (:func:`_build_spec`) and the cache key derive
    from this object, so a constraint cannot change without moving the
    key. The key hashes the canonical dict with keys sorted and
    default-valued fields omitted — adding a field later leaves all
    existing keys in place.
    """

    #: local qubit count
    num_qubits: int
    #: unsigned dual stabilizers in the local basis order
    stabilizers: tuple
    #: per-qubit in-port columns in local order (None = compact 1..n)
    in_columns: tuple = None
    #: per-qubit out-port columns in local order (None = compact 1..n)
    out_columns: tuple = None
    #: stretchable cell: wire-row crossings forbidden between all columns
    clear_wire_row: bool = False

    def canonical(self):
        """The spec as a JSON-safe dict with default-valued fields omitted."""
        data = {"n": self.num_qubits, "stabilizers": list(self.stabilizers)}
        if self.in_columns is not None:
            data["in_columns"] = list(self.in_columns)
        if self.out_columns is not None:
            data["out_columns"] = list(self.out_columns)
        if self.clear_wire_row:
            data["clear_wire_row"] = True
        return data

    def key(self):
        """The cache key: sha256 of the canonical dict."""
        return hashlib.sha256(json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(payload))
    temporary_path.replace(path)


def _dual_stabilizers(local_gates, num_qubits):
    circuit = stim.Circuit()
    gate_map = {"h": "H", "cx": "CX", "x": "X", "s": "S", "sdg": "S_DAG"}
    for name, qubits in local_gates:
        if name not in gate_map:
            raise ValueError(f"synthesize: unsupported gate {name!r} (h/cx/x/s/sdg only)")
        circuit.append(gate_map[name], list(qubits))
    simulator = stim.TableauSimulator()
    simulator.set_num_qubits(num_qubits)
    simulator.do(circuit)
    tableau = simulator.current_inverse_tableau() ** -1

    duals = []
    for qubit in range(num_qubits):
        x_input = [0] * num_qubits
        x_input[qubit] = 1
        duals.append(stim.PauliString(x_input) + tableau.x_output(qubit))
        z_input = [0] * num_qubits
        z_input[qubit] = 3
        duals.append(stim.PauliString(z_input) + tableau.z_output(qubit))
    return duals


def _pauli_parts(pauli):
    text = str(pauli)
    if text.startswith("-"):
        return 1, text[1:]
    if text.startswith("+"):
        return 0, text[1:]
    return 0, text


def _pauli_body_to_bits(body):
    x_bits, z_bits = [], []
    for character in body:
        if character in ("I", "_"):
            x_bits.append(0)
            z_bits.append(0)
        elif character == "X":
            x_bits.append(1)
            z_bits.append(0)
        elif character == "Y":
            x_bits.append(1)
            z_bits.append(1)
        elif character == "Z":
            x_bits.append(0)
            z_bits.append(1)
        else:
            raise ValueError(f"synthesize: unsupported Pauli character {character!r}")
    return x_bits, z_bits


def _bits_to_pauli_body(x_bits, z_bits):
    characters = []
    for x_bit, z_bit in zip(x_bits, z_bits):
        if x_bit == 0 and z_bit == 0:
            characters.append("I")
        elif x_bit == 1 and z_bit == 0:
            characters.append("X")
        elif x_bit == 0 and z_bit == 1:
            characters.append("Z")
        else:
            characters.append("Y")
    return "".join(characters)


def _solve_mod2(matrix, right_hand_side):
    if not matrix:
        return []
    rows = [row[:] for row in matrix]
    values = right_hand_side[:]
    num_rows = len(rows)
    num_columns = len(rows[0])
    pivot_columns = []
    pivot_row = 0
    for column in range(num_columns):
        swap_row = None
        for row in range(pivot_row, num_rows):
            if rows[row][column]:
                swap_row = row
                break
        if swap_row is None:
            continue
        rows[pivot_row], rows[swap_row] = rows[swap_row], rows[pivot_row]
        values[pivot_row], values[swap_row] = values[swap_row], values[pivot_row]
        for row in range(num_rows):
            if row != pivot_row and rows[row][column]:
                for index in range(column, num_columns):
                    rows[row][index] ^= rows[pivot_row][index]
                values[row] ^= values[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == num_rows:
            break

    for row in range(pivot_row, num_rows):
        if values[row]:
            return None

    solution = [0] * num_columns
    for row in range(pivot_row - 1, -1, -1):
        column = pivot_columns[row]
        accumulator = values[row]
        for index in range(column + 1, num_columns):
            accumulator ^= rows[row][index] & solution[index]
        solution[column] = accumulator
    return solution


def _unsigned_stabilizers_and_fixup(duals):
    if not duals:
        return [], ""
    sign_bits, bodies, equations = [], [], []
    width = len(_pauli_parts(duals[0])[1])
    for dual in duals:
        sign_bit, body = _pauli_parts(dual)
        x_bits, z_bits = _pauli_body_to_bits(body)
        sign_bits.append(sign_bit)
        bodies.append(body.replace("_", "I"))
        equations.append(z_bits + x_bits)

    solution = _solve_mod2(equations, sign_bits)
    if solution is None:
        raise RuntimeError("synthesize: failed to solve boundary Pauli fixup system over GF(2)")
    fixup = _bits_to_pauli_body(solution[:width], solution[width:])
    return bodies, fixup


def _build_spec(cell_spec, depth, in_pin_names, out_pin_names):
    num_qubits = cell_spec.num_qubits
    stabilizers = list(cell_spec.stabilizers)
    clear_wire_row = cell_spec.clear_wire_row
    in_columns = list(cell_spec.in_columns) if cell_spec.in_columns is not None else [index + 1 for index in range(num_qubits)]
    out_columns = list(cell_spec.out_columns) if cell_spec.out_columns is not None else [index + 1 for index in range(num_qubits)]

    ports = []
    for index in range(num_qubits):
        ports.append({"location": [in_columns[index], 1, 0], "direction": "+K", "z_basis_direction": "I", "function": in_pin_names[index]})
    for index in range(num_qubits):
        ports.append({"location": [out_columns[index], 1, depth], "direction": "-K", "z_basis_direction": "I", "function": out_pin_names[index]})

    box_width = max(num_qubits + 2, max(list(in_columns) + list(out_columns)) + 2)
    spec = {"max_i": box_width, "max_j": 3, "max_k": depth, "ports": ports, "stabilizers": stabilizers}
    optional = {}
    if clear_wire_row:
        occupied_columns = set(in_columns) | set(out_columns)
        optional["forbidden_cubes"] = [[column, 1, k] for column in range(box_width) if column not in occupied_columns for k in range(depth)]
    if optional:
        spec["optional"] = optional
    return spec


def _hard_wall(signum, frame):
    raise SynthTimeout("synthesize: hard wall — solver attempt exceeded the budget")


def _solve_region(cell_spec, in_pin_names, out_pin_names, min_depth=2, max_depth=16, solver=None, deadline=None):
    num_qubits = cell_spec.num_qubits
    solver = solver if solver is not None else _default_solver()
    synthesizer = _make_synthesizer(solver)
    saved_cap = os.environ.get("MINIFLASH_KISSAT_TIME")

    for depth in range(min_depth, max_depth + 1):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if saved_cap is None:
                    os.environ.pop("MINIFLASH_KISSAT_TIME", None)
                else:
                    os.environ["MINIFLASH_KISSAT_TIME"] = saved_cap
                raise SynthTimeout(f"synthesize: budget exhausted at depth {depth}, n={num_qubits} qubits")
            os.environ["MINIFLASH_KISSAT_TIME"] = str(max(1, int(remaining)))
        spec = _build_spec(cell_spec, depth, in_pin_names, out_pin_names)
        if spec is None:
            continue

        given_vals = None
        if cell_spec.clear_wire_row:
            given_vals = [{"array": "ExistI", "indices": [column, 1, k], "value": 0} for column in range(1, num_qubits) for k in range(depth)]
        try:
            if deadline is not None:
                previous_handler = signal.signal(signal.SIGALRM, _hard_wall)
                signal.setitimer(signal.ITIMER_REAL, max(0.5, remaining))
            solution = synthesizer.solve(specification=spec, given_vals=given_vals, print_detail=False)
        except Exception as error:
            if deadline is not None and time.monotonic() >= deadline:
                raise SynthTimeout("synthesize: hard wall — solver attempt exceeded the budget") from None
            if not (isinstance(error, ValueError) and "return code" in str(error)):
                raise
            solution = None
        finally:
            if deadline is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
                if saved_cap is None:
                    os.environ.pop("MINIFLASH_KISSAT_TIME", None)
                else:
                    os.environ["MINIFLASH_KISSAT_TIME"] = saved_cap
        if solution is None:
            continue

        optimized = solution.after_default_optimizations()
        try:
            verified = optimized.verify_stabilizers_stimzx(spec)
        except Exception as error:
            print(f"synthesize: stimzx verification unavailable ({type(error).__name__}: {error}) — accepting unverified solution", file=sys.stderr)
            verified = None
        if verified is False:
            from lassynth.rewrite_passes.color_z import SEARCH

            decisions = len(SEARCH["made"])
            for trial in range(1, (1 << min(decisions, 5)) if decisions else 1):
                SEARCH["plan"] = tuple((trial >> bit) & 1 for bit in range(decisions))[::-1]
                optimized = solution.after_default_optimizations()
                try:
                    verified = optimized.verify_stabilizers_stimzx(spec)
                except Exception:
                    verified = None
                if verified is not False:
                    break
            SEARCH["plan"] = ()
        if verified is False:
            print(f"synthesize: stimzx verification FAILED at depth {depth} — retrying deeper", file=sys.stderr)
            continue
        return optimized.lasre

    raise SynthUnsat(f"synthesize: no solution for depths {min_depth}..{max_depth} with {solver}, n={num_qubits} qubits, stabilizers={list(cell_spec.stabilizers)}")


def synthesize_region(region, cache_dir=".miniflash-cache", in_columns=None, out_columns=None, in_bases=None, budget_s=None, use_sat=True, home_columns=None) -> Cell:
    """SAT-synthesize (or load from disk cache) the lattice-surgery cell for one region.

    With ``home_columns`` the cell is solved in canonical form: local
    qubits ordered by home column, compact port columns, wire-row
    crossings forbidden between all columns, pins named by column rank.
    The ``canon.v1`` entry is therefore independent of the permutation
    and of the spectator gaps it is stretched over on return; pins are
    renamed to the region's own numbering at load. Without it, ports
    stay pinned as given (``free.v1``).

    :param region: Region.
    :param cache_dir: str, disk cache directory.
    :param in_columns: {qubit: local column} in-pin constraints.
    :param out_columns: {qubit: local column} out-pin constraints.
    :param in_bases: {qubit: 0|1} wire ledger bits at entry, recorded on the
        cell for the Hadamard ledger; the geometry is solved for the
        region's gates alone (the channel landing pays any owed Hadamard).
    :param use_sat: bool, run the solver on a cache miss; False raises
        SynthTimeout instead, so the caller's fallback takes over.
    :param home_columns: {qubit: home column} for fixed-columns mode.
    :returns: Cell (pauli_frame set to the sign fixup).
    :raises SynthTimeout: on budget exhaustion (also when a .synthfail marker
        records a timeout at >= this budget by an equal-or-stronger solver,
        or on any miss with SAT disabled).
    :raises RuntimeError: on UNSAT or failed verification.
    """
    order, column_of = list(region.qubits), None
    if home_columns is not None:
        order = sorted(region.qubits, key=home_columns.__getitem__)
        homes = [home_columns[qubit] for qubit in order]
        column_of = [homes[0] - 1] + homes + [homes[-1] + 1]
        in_columns = out_columns = None

    num_qubits = len(region.qubits)
    local_index_of = {qubit: index for index, qubit in enumerate(order)}
    local_gates = _local_gates(region, order)
    stabilizers, fixup = _unsigned_stabilizers_and_fixup(_dual_stabilizers(local_gates, num_qubits))
    spec = CellSpec(
        num_qubits=num_qubits,
        stabilizers=tuple(stabilizers),
        in_columns=tuple(in_columns.get(qubit, 1) for qubit in order) if in_columns is not None else None,
        out_columns=tuple(out_columns.get(qubit, 1) for qubit in order) if out_columns is not None else None,
        clear_wire_row=home_columns is not None,
    )
    key = spec.key()
    regime_root = Path(cache_dir) / (CACHE_CANON if home_columns is not None else CACHE_FREE)
    cache_path = regime_root / f"{key}.json"
    fail_path = regime_root / f"{key}.synthfail"

    def instance(cell):
        cell.pauli_frame = fixup
        cell.in_bases = dict(in_bases) if in_bases else {}
        if home_columns is None:
            return cell
        for pin in cell.pins:
            prefix, _, index = pin["name"].rpartition("_")
            qubit = order[int(index)]
            pin["name"] = region.in_pins[qubit] if prefix == "in" else region.out_pins[qubit]
        return _map_columns(cell, column_of)

    if cache_path.exists():
        return instance(Cell.from_dict(json.loads(cache_path.read_text())))
    if not use_sat:
        raise SynthTimeout(f"synthesize: cache miss with SAT disabled, n={num_qubits} qubits")

    solver = _default_solver()
    if budget_s and fail_path.exists():
        marker = json.loads(fail_path.read_text())
        if marker["budget"] >= budget_s and marker["solver"] in ("kissat", solver):
            raise SynthTimeout(f"synthesize: cached timeout at budget {marker['budget']:.0f}s with {marker['solver']}, n={num_qubits} qubits")

    in_pin_names = [f"in_{index}" for index in range(num_qubits)] if home_columns is not None else [region.in_pins[qubit] for qubit in order]
    out_pin_names = [f"out_{index}" for index in range(num_qubits)] if home_columns is not None else [region.out_pins[qubit] for qubit in order]
    deadline = time.monotonic() + budget_s if budget_s else None

    try:
        lasre = _solve_region(spec, in_pin_names, out_pin_names, solver=solver, deadline=deadline)
    except SynthTimeout:
        if budget_s:
            _write_json(fail_path, {"budget": budget_s, "solver": solver})
        raise

    cell = from_lasre(lasre, name="region")
    payload = cell.to_dict()
    payload["spec"] = spec.canonical()
    _write_json(cache_path, payload)
    return instance(cell)


ENGINES = ("auto", "sat", "packing")


def _packed_cell(region, in_columns, out_columns, in_bases, tracks=None):
    from .packing import pack_cell

    if in_columns != out_columns:
        raise SynthUnsat("synthesize: the packer keeps every qubit in its entry column")
    try:
        return pack_cell(region, in_columns=in_columns, in_bases=in_bases, tracks=tracks)
    except (ValueError, RuntimeError) as unsupported:
        raise SynthUnsat(f"synthesize: {unsupported}") from None


MERGE_CAP = 8
#: permutation cost/search algorithm version — bump on change, the signature only sees inputs
PERM_ALGO_VERSION = 1


def _merge_interleaved(layer_regions, column_of):
    from .partition import Region

    boxes = sorted(layer_regions, key=lambda region: min(column_of[qubit] for qubit in region.qubits))
    groups, group_high = [], None
    for region in boxes:
        low = min(column_of[qubit] for qubit in region.qubits)
        high = max(column_of[qubit] for qubit in region.qubits)
        if groups and low <= group_high:
            groups[-1].append(region)
            group_high = max(group_high, high)
        else:
            groups.append([region])
            group_high = high
    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        picked = sorted((index, gate) for region in group for index, gate in zip(region.gate_indices, region.gates))
        qubits = sorted({qubit for region in group for qubit in region.qubits})
        joint = Region(qubits=qubits, gates=[gate for _, gate in picked],
                       in_pins={qubit: f"in_{local}" for local, qubit in enumerate(qubits)},
                       out_pins={qubit: f"out_{local}" for local, qubit in enumerate(qubits)},
                       first_gate_index=min(region.first_gate_index for region in group),
                       gate_indices=[index for index, _ in picked])
        joint.merge_keys = frozenset(region.gate_indices[0] for region in group)
        merged.append(joint)
    return merged


def _fixed_permutation(partitioned, seed=0, iterations=6000, no_merge=(), cache_dir=None):
    import random

    cache_path = None
    if cache_dir is not None:
        signature = (partitioned.num_qubits,
                     tuple((region.first_gate_index, tuple(region.qubits), tuple(region.gate_indices)) for region in partitioned.regions),
                     tuple((event.qubit, event.index) for event in partitioned.events),
                     seed, iterations, MERGE_CAP, tuple(sorted(tuple(sorted(banned)) for banned in no_merge)))
        key = hashlib.sha256(repr((PERM_ALGO_VERSION,) + signature).encode()).hexdigest()
        cache_path = Path(cache_dir) / CACHE_PERMS / f"{key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

    rng = random.Random(seed)
    num_qubits = partitioned.num_qubits
    adjacency = [[0] * num_qubits for _ in range(num_qubits)]
    for region in partitioned.regions:
        for a in region.qubits:
            for b in region.qubits:
                if a != b:
                    adjacency[a][b] += 1

    def cost(perm):
        layers, _ = schedule_layers(partitioned, disjoint_columns=True, column_of=perm, merge_cap=MERGE_CAP, no_merge=no_merge)
        spans = sum(max(perm[q] for q in region.qubits) - min(perm[q] for q in region.qubits) for layer in layers for region in layer)
        return (len(layers), spans)

    placed, remaining = [], set(range(num_qubits))
    current = min(remaining, key=lambda q: sum(adjacency[q]))
    while True:
        placed.append(current)
        remaining.discard(current)
        if not remaining:
            break
        current = max(remaining, key=lambda q: sum(adjacency[q][p] for p in placed[-4:]))
    greedy = [0] * num_qubits
    for position, qubit in enumerate(placed):
        greedy[qubit] = position

    column = min((list(range(num_qubits)), greedy), key=cost)
    current_cost = cost(column)
    for _ in range(iterations):
        order = sorted(range(num_qubits), key=lambda q: column[q])
        source = order.pop(rng.randrange(num_qubits))
        order.insert(rng.randrange(num_qubits), source)
        candidate = [0] * num_qubits
        for position, qubit in enumerate(order):
            candidate[qubit] = position
        trial = cost(candidate)
        if trial <= current_cost:
            column, current_cost = candidate, trial
    if cache_path is not None:
        _write_json(cache_path, column)
    return column


def _fixed_floorplan(layers, num_qubits, column_of):
    from .floorplan import Floorplan

    num_layers = len(layers)
    base_width = num_qubits + 2
    in_port_columns, out_port_columns, lane_columns, placements = [], [], [], []
    for boxes in layers:
        union = {qubit for region in boxes for qubit in region.qubits}
        columns = {qubit: column_of[qubit] + 1 for qubit in union}
        in_port_columns.append(dict(columns))
        out_port_columns.append(dict(columns))
        lane_columns.append({qubit: column_of[qubit] + 1 for qubit in range(num_qubits) if qubit not in union})
        placements.append([(0, 0, list(region.qubits)) for region in boxes])
    return Floorplan(num_qubits=num_qubits, box_widths=[base_width] * num_layers, in_port_columns=in_port_columns, out_port_columns=out_port_columns, lane_columns=lane_columns, moves=[[] for _ in range(max(0, num_layers - 1))], gap_levels=[0] * max(0, num_layers - 1), placements=placements)


def synthesize(partitioned, cache_dir=".miniflash-cache", die_dims=None, budget_s=None, engine="auto", fixed_columns=False, no_merge=()):
    """Run the synthesis stage for a whole circuit: schedule, floorplan, then every cell.

    Threads the wire parity ledger between cells (a cell's out-pin parities become
    the next consumer's in_bases, and a set bit lands the channel's flip wall before
    that consumer); failed regions are tagged on the raised error (``error.region``)
    so callers can split and retry.

    :param partitioned: PartitionedCircuit.
    :param cache_dir: str, cell cache directory.
    :param die_dims: (width, rows | None) or None for 1-D.
    :param budget_s: float | None, SAT budget in seconds per region.
    :param engine: "auto" (default) serves cache hits and packs the rest with
        :func:`miniflash.packing.pack_cell`; "sat" never packs, solving every
        miss and letting the caller split what the solver cannot take;
        "packing" never reads the cache, so its result does not depend on what
        happens to be on disk.
    :param no_merge: iterable of frozensets, interleave components (by first
        gate index) that must not fuse into a joint cell — failed merges the
        caller is retrying without.

    Under ``engine="auto"`` a region whose cache lookup misses falls to
    :func:`miniflash.packing.pack_cell` — plane-level packing on two tracks,
    no solver — keeping the region one cell in one layer; the error propagates
    only when the packer has no shape for it (reordered out columns, a gate
    without a pattern).
    :returns: (floorplan, cell_types, channels) — everything elaborate consumes.
    :raises SynthTimeout | SynthUnsat: from the failing region, with ``.region`` set.
    """
    regions, events, num_qubits = partitioned.regions, partitioned.events, partitioned.num_qubits
    column_of = _fixed_permutation(partitioned, no_merge=no_merge, cache_dir=cache_dir) if fixed_columns else None
    layers, channels = schedule_layers(partitioned, die_dims=die_dims, disjoint_columns=fixed_columns, column_of=column_of, merge_cap=MERGE_CAP if fixed_columns else None, no_merge=no_merge)

    if regions and fixed_columns:
        layers = [_merge_interleaved(layer, column_of) for layer in layers]
        floorplan = _fixed_floorplan(layers, num_qubits, column_of)
    elif regions:
        floorplan = solve_floorplan([[region.qubits for region in layer] for layer in layers], num_qubits, die_dims=die_dims)
    else:
        floorplan = passthrough_floorplan(num_qubits)

    cell_types = []
    wire_bits = {qubit: 0 for qubit in range(num_qubits)}
    for layer, layer_regions in enumerate(layers):
        region_of = {tuple(region.qubits): region for region in layer_regions}
        layer_cells, ledger_boxes = [], []
        for _, offset, box_qubits in floorplan.placements[layer]:
            region = region_of[tuple(box_qubits)]

            in_columns = {qubit: floorplan.in_port_columns[layer][qubit] - offset for qubit in region.qubits if qubit in floorplan.in_port_columns[layer]}
            out_columns = {qubit: floorplan.out_port_columns[layer][qubit] - offset for qubit in region.qubits if qubit in floorplan.out_port_columns[layer]}
            in_bases = {qubit: wire_bits[qubit] for qubit in region.qubits}
            tracks = 2 if die_dims is not None else None
            if engine == "packing":
                try:
                    cell = _packed_cell(region, in_columns, out_columns, in_bases, tracks)
                except SynthUnsat as error:
                    error.region = region
                    raise
            else:
                try:
                    if fixed_columns:
                        cell = synthesize_region(region, cache_dir=str(cache_dir), in_bases=in_bases, budget_s=budget_s, use_sat=engine == "sat", home_columns=in_columns)
                    else:
                        cell = synthesize_region(region, cache_dir=str(cache_dir), in_columns=in_columns, out_columns=out_columns, in_bases=in_bases, budget_s=budget_s, use_sat=engine == "sat")
                except (SynthTimeout, SynthUnsat) as error:
                    if engine == "sat":
                        error.region = region
                        error.merge_keys = getattr(region, "merge_keys", None)
                        raise error
                    try:
                        cell = _packed_cell(region, in_columns, out_columns, in_bases, tracks)
                    except SynthUnsat:
                        error.region = region
                        raise error

            layer_cells.append(cell)
            ledger_boxes.append(out_columns)

        for cell, out_columns in zip(layer_cells, ledger_boxes):
            wire_bits.update(cell.pin_parities(out_columns, "out"))
        cell_types.append(layer_cells)

    return floorplan, cell_types, channels
