# Windows, slabs and cells in 3-D. window_box gives the cube corners of a window on the glTF lattice, boxes one box per
# window coloured by slab for write_gltf's overlays, timeline where every step and slab starts once the cells are in,
# lower_cells the pipes of a program whose slabs are replaced by cells, and write_cells that scene as a glTF file. The
# overlay view keeps every pipe and lays the boxes over them, while the carve view removes what is inside the boxes so
# that each window stands as an opaque cell, the program after coarsening and before any cell is resynthesised.
from __future__ import annotations

from typing import Iterable, Sequence

from .. import constant
from ..gltf import _Pipe, _merge_pipes, _qubit_pipes, write_scene
from ..program import MAGIC, Program
from .cell import Cell
from .window import Slab, Window

# Colours the pipe materials do not use, which are red, blue, grey, yellow, green and orange.
PALETTE = ((0.55, 0.25, 0.80), (0.10, 0.65, 0.70), (0.85, 0.30, 0.60),
           (0.45, 0.60, 0.10), (0.60, 0.40, 0.20), (0.20, 0.45, 0.85))
Y_CUBE = (0.90, 0.25, 0.65, 1.0)   # pink
Plan = Sequence[tuple[Slab, Sequence[Cell]]]


def window_box(window: Window) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    # Inclusive (low, high) cube corners, columns along I, rows along J and both layers of every step along K.
    stride = constant.GLTF_STRIDE
    low = (stride * window.box.c0, stride * window.box.r0, stride * constant.LAYERS_PER_STEP * window.start)
    high = (stride * window.box.c1, stride * window.box.r1, stride * (constant.LAYERS_PER_STEP * window.stop - 1))
    return low, high


def boxes(slabs: Iterable[Slab], alpha: float = constant.WINDOW_ALPHA) -> list[tuple[tuple, tuple, tuple]]:
    # One (low, high, rgba) per window, the windows of one slab sharing a colour and consecutive slabs cycling PALETTE.
    out = []
    for slab_index, slab in enumerate(slabs):
        rgb = PALETTE[slab_index % len(PALETTE)]
        for window in slab.windows:
            low, high = window_box(window)
            out.append((low, high, rgb + (alpha,)))
    return out


def slab_depth(slab: Slab, cells: Sequence[Cell], compact: bool = True) -> int:
    # Layers the slab takes, its deepest cell, or its own two per step when not compact.
    if not compact:
        return constant.LAYERS_PER_STEP * slab.steps
    return max((cell.depth for cell in cells), default=constant.LAYERS_PER_STEP * slab.steps)


def timeline(program: Program, plan: Plan, compact: bool = True) -> tuple[dict[int, int], int]:
    # A step inside a slab maps to the slab's first layer, and a step outside every slab keeps its two layers.
    by_start = {slab.start: (slab, cells) for slab, cells in plan}
    base = {}
    layer = 0
    step_index = 0
    while step_index < len(program.steps):
        if step_index in by_start:
            slab, cells = by_start[step_index]
            for covered in range(slab.start, slab.stop):
                base[covered] = layer
            layer += slab_depth(slab, cells, compact)
            step_index = slab.stop
        else:
            base[step_index] = layer
            layer += constant.LAYERS_PER_STEP
            step_index += 1
    return base, layer   # {step: first layer}, total layers


def lower_cells(program: Program, plan: Plan, compact: bool = True,
                outline: float = constant.CELL_OUTLINE) -> tuple[list[_Pipe], list[tuple], list[tuple], list[tuple], int]:
    # With compact every slab is as deep as its deepest cell, without it a shallow cell leaves its qubits idle above it.
    stride = constant.GLTF_STRIDE
    base, total = timeline(program, plan, compact)
    found: dict[tuple[tuple, tuple], _Pipe] = {}
    magic = set()
    overlays = []

    def cube(tile, layer):
        return (stride * tile[1], stride * tile[0], stride * layer)

    def put(start, end, parity, hadamard=False, walk=False):
        low, high = min(start, end), max(start, end)
        if low[2] < 0 or high[2] > stride * (total - 1):
            return
        axis = "IJK"[next(i for i in range(3) if low[i] != high[i])]
        found[(low, high)] = _Pipe(low, high, axis, parity, hadamard=hadamard, walk=walk)

    covered = {step_index for slab, _ in plan for step_index in range(slab.start, slab.stop)}
    steps = program.steps
    for step_index, step in enumerate(steps):
        if step_index in covered:
            continue
        after = steps[step_index + 1] if step_index + 1 < len(steps) else None
        move = {2 * step_index: base[step_index], 2 * step_index + 1: base[step_index] + 1,
                2 * step_index + 2: base.get(step_index + 1, total)}
        for pipe in _merge_pipes(step, step_index) + _qubit_pipes(step, step_index, after):
            start = pipe.lo[:2] + (stride * move[pipe.lo[2] // stride],)
            end = pipe.hi[:2] + (stride * move[pipe.hi[2] // stride],)
            put(start, end, pipe.parity, walk=pipe.walk)
        for vertex in step.of_kind(MAGIC):
            if vertex.id >= 0:
                magic.add(cube(vertex.tile, base[step_index]))

    for slab_index, (slab, cells) in enumerate(plan):
        first = base[slab.start]
        depth = slab_depth(slab, cells, compact)
        seats = steps[slab.start].qubit_tiles()
        busy = {qubit for cell in cells for qubit in cell.window.participants}
        for qubit, tile in seats.items():
            if qubit not in busy:
                for layer in range(first, first + depth):
                    put(cube(tile, layer), cube(tile, layer + 1), 0)

        for cell in cells:
            box = cell.window.box

            def at(ijk, box=box):
                return (stride * (box.c0 + ijk[0]), stride * (box.r0 + ijk[1]), stride * (first + ijk[2] - 1))

            for pipe in cell.pipes():
                offset = tuple(int(axis == pipe.axis) for axis in "IJK")
                successor = tuple(coordinate + delta for coordinate, delta in zip(pipe.cube, offset))
                put(at(pipe.cube), at(successor), pipe.parity, hadamard=pipe.wall)
            for qubit in cell.window.participants:
                for layer in range(first + cell.depth, first + depth):
                    put(cube(seats[qubit], layer), cube(seats[qubit], layer + 1), 0)
            for magic_cube in cell.magic_cubes():
                magic.add(at(magic_cube))
            for y_cube in cell.y_cubes():
                overlays.append((at(y_cube), at(y_cube), Y_CUBE))
            if outline > 0:   # a translucent box around the cell, outline being its alpha
                low = at((0, 0, 1))
                high = at((box.width - 1, box.height - 1, cell.depth))
                overlays.append((low, high, PALETTE[slab_index % len(PALETTE)] + (outline,)))

    pipes = sorted(found.values(), key=lambda pipe: (pipe.lo, pipe.hi, pipe.axis))
    cubes = {pipe.lo for pipe in pipes} | {pipe.hi for pipe in pipes}
    cubes -= magic
    cubes -= {low for low, high, _rgba in overlays if low == high}
    return pipes, sorted(cubes), sorted(magic), overlays, total


def write_cells(program: Program, plan: Plan, path, compact: bool = True, outline: float = constant.CELL_OUTLINE) -> int:
    # Returns the layers the scene takes.
    pipes, cubes, magic, overlays, total = lower_cells(program, plan, compact, outline)
    write_scene(path, pipes, cubes, magic, overlays=overlays)
    return total
