# The glTF backend lowers a tile Program to pipes on a cube lattice and writes a glTF 2.0 scene. Every step spans two
# layers, 2t and 2t + 1, and tile (row, col) on layer k sits at cube (STRIDE * col, STRIDE * row, STRIDE * k). Each
# qubit is a red K pipe through every layer, and a walk adds its I and J pipes in orange on the first layer of the
# later step. A CX merge leaves its control vertically on the lower layer in blue, climbs at its first turn, the
# ancilla, and enters its target horizontally on the upper layer in red, while a T merge lies on the lower layer in
# blue. Grey cubes mark every occupied tile and green marks the consumed magic tiles, the factories and the deliveries.
from __future__ import annotations

import base64
import json
import struct
from typing import Iterable, NamedTuple, Optional

from . import constant
from .mapping import Coord
from .program import MAGIC, QUBIT, ROUTE, Program, Step
from .supply import Supply


_UNIT_POSITIONS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
_UNIT_INDICES = (0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 5, 1, 0, 4, 5, 3, 2, 6, 3, 6, 7, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2)

Cube = tuple[int, int, int]


class _Pipe(NamedTuple):
    lo: Cube
    hi: Cube
    axis: str                # I, J or K
    parity: int              # 0 red for X, 1 blue for Z
    hadamard: bool = False   # a Hadamard wall on this segment
    t_volume: bool = False   # magic-state volume
    walk: bool = False       # a qubit moving between partitions


def _cube(tile: Coord, layer: int) -> Cube:
    return (constant.GLTF_STRIDE * tile[1], constant.GLTF_STRIDE * tile[0], constant.GLTF_STRIDE * layer)


def _pipe(start: Cube, end: Cube, parity: int) -> _Pipe:
    # The two cubes differ on exactly one axis.
    low, high = min(start, end), max(start, end)
    axis = "IJK"[next(i for i in range(3) if low[i] != high[i])]
    return _Pipe(low, high, axis, parity)


def _route_path(step: Step, component: Iterable[Coord], z_tile: Coord) -> list[Coord]:
    # Tiles of a merge in walking order from the Z party to the X party.
    neighbours = step.neighbours()
    path = [z_tile]
    previous = None
    while True:
        onward = [tile for tile in neighbours[path[-1]] if tile != previous and tile in component]
        if not onward or (len(path) > 1 and step.vertices[path[-1]].kind != ROUTE):
            return path
        previous = path[-1]
        path.append(onward[0])


def _z_party(step: Step, component: Iterable[Coord]) -> Coord:
    # The qubit merged through its top or bottom, the control of a CX or the qubit of a T.
    neighbours = step.neighbours()
    for tile in component:
        if step.vertices[tile].kind == QUBIT and all(neighbour[1] == tile[1] for neighbour in neighbours[tile]):
            return tile


def _direction(src: Coord, dst: Coord) -> tuple[int, int]:
    return (dst[0] - src[0], dst[1] - src[1])


def ancilla(path: list[Coord]) -> int:
    # Index of the first tile where the route turns.
    for i in range(1, len(path) - 1):
        if _direction(path[i - 1], path[i]) != _direction(path[i], path[i + 1]):
            return i


def _merge_pipes(step: Step, step_index: int) -> list[_Pipe]:
    # Z halves and T merges lie on the lower layer of the step, the ancilla climbs, and X halves lie on the upper one.
    lower_layer, upper_layer = 2 * step_index, 2 * step_index + 1
    pipes = []
    for component in step.components():
        path = _route_path(step, component, _z_party(step, component))
        if step.vertices[path[-1]].kind == MAGIC:
            pipes.extend(_pipe(_cube(tile_a, lower_layer), _cube(tile_b, lower_layer), 1)
                         for tile_a, tile_b in zip(path, path[1:]))
            continue
        turn = ancilla(path)
        pipes.extend(_pipe(_cube(tile_a, lower_layer), _cube(tile_b, lower_layer), 1)
                     for tile_a, tile_b in zip(path[:turn], path[1:turn + 1]))
        pipes.append(_pipe(_cube(path[turn], lower_layer), _cube(path[turn], upper_layer), 1))
        pipes.extend(_pipe(_cube(tile_a, upper_layer), _cube(tile_b, upper_layer), 0)
                     for tile_a, tile_b in zip(path[turn:], path[turn + 1:]))
    return pipes


def _qubit_pipes(step: Step, step_index: int, after: Optional[Step] = None) -> list[_Pipe]:
    # Each qubit's K pipe through the step and on to the next step, with its walk on that step's first layer.
    pipes = []
    seats = step.qubit_tiles()
    next_seats = after.qubit_tiles() if after is not None else {}
    for qubit, src in seats.items():
        pipes.append(_pipe(_cube(src, 2 * step_index), _cube(src, 2 * step_index + 1), 0))
        if after is None:
            continue
        dst = next_seats[qubit]
        next_layer = 2 * step_index + 2
        pipes.append(_pipe(_cube(src, 2 * step_index + 1), _cube(src, next_layer), 0))
        corner = (src[0], dst[1])
        if corner != src:
            pipes.append(_pipe(_cube(src, next_layer), _cube(corner, next_layer), 0)._replace(walk=True))
        if dst != corner:
            pipes.append(_pipe(_cube(corner, next_layer), _cube(dst, next_layer), 0)._replace(walk=True))
    return pipes


def _delivery_pipes(supply: Supply) -> list[_Pipe]:
    # Green pipes along every delivery path, on the lower layer of its step.
    pipes = []
    for delivery in supply.deliveries:
        layer = 2 * delivery.step
        pipes.extend(_pipe(_cube(tile_a, layer), _cube(tile_b, layer), 1)._replace(t_volume=True)
                     for tile_a, tile_b in zip(delivery.tiles, delivery.tiles[1:]))
    return pipes


def factory_boxes(program: Program, supply: Supply) -> list[tuple[Cube, Cube]]:
    # The (low, high) corners of each factory box through every layer of the program, high exclusive.
    stride = constant.GLTF_STRIDE
    top = stride * (2 * len(program.steps) - 1) + 1
    return [((stride * box.col, stride * box.row, 0),
             (stride * (box.col + box.width - 1) + 1, stride * (box.row + box.height - 1) + 1, top))
            for box in supply.boxes]


def lower(program: Program, supply: Optional[Supply] = None) -> tuple[list[_Pipe], list[Cube], list[Cube]]:
    # A supply adds the delivery pipes, and its factory ports are left out of the cubes since they sit in the boxes.
    pipes = []
    cubes = set()
    magic = set()
    steps = program.steps
    for step_index, step in enumerate(steps):
        for vertex in step.of_kind(MAGIC):
            if vertex.id >= 0:
                magic.add(_cube(vertex.tile, 2 * step_index))
        pipes.extend(_merge_pipes(step, step_index))
        pipes.extend(_qubit_pipes(step, step_index, steps[step_index + 1] if step_index + 1 < len(steps) else None))
    if supply is not None:
        pipes.extend(_delivery_pipes(supply))

    for pipe in pipes:
        cubes.add(pipe.lo)
        cubes.add(pipe.hi)
    cubes -= magic
    if supply is not None:
        cubes -= {_cube(supply.boxes[delivery.factory].port, 2 * delivery.step) for delivery in supply.deliveries}
    pipes.sort(key=lambda pipe: (pipe.lo, pipe.hi, pipe.axis))
    return pipes, sorted(cubes), sorted(magic)   # pipes, grey cubes, consumed magic cubes


def _collect_pipe_boxes(pipes: list[_Pipe]) -> list[tuple[Cube, int, str]]:
    entries = []
    for pipe in pipes:
        low, high = tuple(pipe.lo), tuple(pipe.hi)
        middle = tuple((low[i] + high[i]) // 2 for i in range(3))
        if pipe.walk:
            material = 5
        elif pipe.hadamard:
            material = 3
        elif pipe.t_volume:
            material = 4
        else:
            material = 1 if pipe.parity >= 1 else 0
        entries.append((low, material, middle, pipe.axis))
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    return [(middle, material, axis) for _low, material, middle, axis in entries]


def _pipe_node(middle: Cube, material: int, axis: str) -> dict:
    axis_index = {"I": 0, "J": 1, "K": 2}[axis]
    scale = [constant.PIPE_CROSS_SCALE] * 3
    scale[axis_index] = constant.PIPE_AXIS_SCALE
    margin = (1.0 - constant.PIPE_CROSS_SCALE) / 2
    translation = [float(coordinate) + margin for coordinate in middle]
    translation[axis_index] = float(middle[axis_index]) - (constant.PIPE_AXIS_SCALE - 1.0) / 2
    return {"mesh": material, "translation": translation, "scale": scale}


def _pack_shared_mesh() -> tuple[bytes, int, int, int]:
    position_bytes = struct.pack("<24f", *(value for vertex in _UNIT_POSITIONS for value in vertex))
    index_bytes = struct.pack("<36H", *_UNIT_INDICES)
    buffer = bytearray(position_bytes)
    index_offset = len(buffer)
    buffer += index_bytes
    while len(buffer) % 4 != 0:
        buffer.append(0)
    return bytes(buffer), len(position_bytes), index_offset, len(index_bytes)


def _inside(position: Cube, low: Cube, high: Cube) -> bool:
    return all(low[i] <= position[i] <= high[i] for i in range(3))


def write_gltf(program: Program, path, supply: Optional[Supply] = None, overlays: Iterable[tuple] = (),
               carve: bool = False) -> None:
    # carve drops every pipe and cube inside an overlay, a (low, high, rgba) box of inclusive cube corners.
    pipes, cubes, magic = lower(program, supply)
    boxes = factory_boxes(program, supply) if supply is not None else []
    overlays = [(tuple(low), tuple(high), tuple(rgba)) for low, high, rgba in overlays]
    if carve and overlays:
        def outside(*positions):
            return not any(all(_inside(position, low, high) for position in positions) for low, high, _ in overlays)
        pipes = [pipe for pipe in pipes if outside(pipe.lo, pipe.hi)]
        cubes = [cube for cube in cubes if outside(cube)]
        magic = [magic_cube for magic_cube in magic if outside(magic_cube)]
    write_scene(path, pipes, cubes, magic, boxes, overlays)


def write_scene(path, pipes: list[_Pipe], cubes: Iterable[Cube], magic: Iterable[Cube], boxes: Iterable[tuple] = (),
                overlays: Iterable[tuple] = ()) -> None:
    # Factory boxes take high exclusive and overlays inclusive, each overlay colour becoming its own material.
    overlays = [(tuple(low), tuple(high), tuple(rgba)) for low, high, rgba in overlays]
    pipe_boxes = _collect_pipe_boxes(pipes)
    buffer, position_bytes_length, index_offset, index_bytes_length = _pack_shared_mesh()

    root = {
        "asset": {"version": "2.0", "generator": "miniflash"},
        "buffers": [{"byteLength": len(buffer), "uri": "data:application/octet-stream;base64," + base64.b64encode(buffer).decode("ascii")}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": position_bytes_length, "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": index_bytes_length, "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 8, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
            {"bufferView": 1, "byteOffset": 0, "componentType": 5123, "count": 36, "type": "SCALAR"},
        ],
        "materials": [
            {"pbrMetallicRoughness": {"baseColorFactor": [0.85, 0.15, 0.15, 1.0]}},
            {"pbrMetallicRoughness": {"baseColorFactor": [0.15, 0.15, 0.85, 1.0]}},
            {"pbrMetallicRoughness": {"baseColorFactor": [0.55, 0.55, 0.55, 1.0]}},
            {"pbrMetallicRoughness": {"baseColorFactor": [0.95, 0.78, 0.10, 1.0]}},
            {"pbrMetallicRoughness": {"baseColorFactor": [0.15, 0.75, 0.25, 1.0]}},
            {"pbrMetallicRoughness": {"baseColorFactor": [1.00, 0.50, 0.05, 1.0]}},
        ],
    }
    palette = list(dict.fromkeys(rgba for _, _, rgba in overlays))
    for rgba in palette:
        material = {"pbrMetallicRoughness": {"baseColorFactor": [float(channel) for channel in rgba]}, "doubleSided": True}
        if rgba[3] < 1.0:
            material["alphaMode"] = "BLEND"
        root["materials"].append(material)
    root["meshes"] = [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": material}]}
                      for material in range(len(root["materials"]))]

    nodes = []
    for middle, material, axis in pipe_boxes:
        nodes.append(_pipe_node(middle, material, axis))
    for position in cubes:
        nodes.append({"mesh": 2, "translation": [float(coordinate) for coordinate in position], "scale": [1.0, 1.0, 1.0]})
    for position in magic:
        nodes.append({"mesh": 4, "translation": [float(coordinate) for coordinate in position], "scale": [1.0, 1.0, 1.0]})
    for low, high in boxes:
        nodes.append({"mesh": 4, "translation": [float(coordinate) for coordinate in low],
                      "scale": [float(high[i] - low[i]) for i in range(3)]})
    for low, high, rgba in overlays:
        nodes.append({"mesh": 6 + palette.index(rgba), "translation": [float(coordinate) for coordinate in low],
                      "scale": [float(high[i] - low[i] + 1) for i in range(3)]})

    # The root node turns the scene -90 degrees about X so that K, the time axis, points up and J points into the screen.
    nodes.append({"children": list(range(len(nodes))), "rotation": [-0.7071068, 0.0, 0.0, 0.7071068]})
    root["nodes"] = nodes
    root["scenes"] = [{"nodes": [len(nodes) - 1]}]
    root["scene"] = 0

    with open(path, "w") as file:
        json.dump(root, file, indent=2)
        file.write("\n")
