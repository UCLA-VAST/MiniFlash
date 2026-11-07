# A closed window as a lassynth problem. The window's box becomes the lassynth volume, one cube per tile and layer,
# with columns along I, rows along J and layers along K. Layer 0 holds nothing but the open ends of the input ports, so
# a window of L layers asks for max_k = L + 1 and the tile program's layer l sits at k = l + 1. Each participant enters
# through the bottom of its tile and leaves through the top as a K pipe, and each magic state consumed is a port on its
# magic tile at the layer of the merge, facing the route that reaches it. The stabilizers are the window's pairs in
# execution order on a Choi state, a T merge being a Z spider on its qubit with one leg to the magic port. The ports,
# which are the stabilizer columns, run inputs by qubit, outputs by qubit, then the magic ports in execution order.
from __future__ import annotations

from typing import Optional

from .. import constant
from ..circuit import Problem
from ..gltf import _route_path, _z_party, ancilla
from ..mapping import Coord
from ..program import MAGIC, Program
from .window import Component, Window

QUBIT_Z = "I"       # lassynth names the faces the other way round from the tile model, so a qubit's Z side is I there
MAGIC_Z = "plane"   # a magic port's Z boundary faces the in-plane axis across its pipe, or time with "K"


def layers(window: Window) -> int:
    return constant.LAYERS_PER_STEP * window.steps


def port_names(window: Window) -> list[str]:
    qubits = window.participants
    return ([f"in q{qubit}" for qubit in qubits] + [f"out q{qubit}" for qubit in qubits]
            + [f"magic p{component.pair}" for component in window.components if component.magic is not None])


def stabilizers(window: Window, problem: Problem) -> list[str]:
    # Unsigned stabilizer generators over the ports, in canonical form.
    import stim
    qubits = window.participants
    n_qubits = len(qubits)
    wire = {qubit: n_qubits + i for i, qubit in enumerate(qubits)}
    circuit = stim.Circuit()
    for i in range(n_qubits):
        circuit.append("H", [i])
        circuit.append("CX", [i, n_qubits + i])

    n_magic = 0
    for component in window.components:
        pair = problem.pairs[component.pair]
        if pair.is_magic:
            circuit.append("CX", [wire[pair.z], 2 * n_qubits + n_magic])
            n_magic += 1
        else:
            circuit.append("CX", [wire[pair.z], wire[pair.x]])

    simulator = stim.TableauSimulator()
    simulator.set_num_qubits(2 * n_qubits + n_magic)
    simulator.do(circuit)
    return [str(stabilizer)[1:].replace("_", "I") for stabilizer in simulator.canonical_stabilizers()]


def _cube(window: Window, tile: Coord, layer: int) -> tuple[int, int, int]:
    return (tile[1] - window.box.c0, tile[0] - window.box.r0, layer)


def _seats(window: Window, program: Program) -> dict[int, Coord]:
    return program.steps[window.start].qubit_tiles()


def _magic_ports(window: Window, program: Program, depth: int) -> list[tuple[Component, int, Coord]]:
    # Each magic port sits on the lower layer of its step, lowered to fit depth and to keep two uses of one tile apart.
    taken = set()
    ports = []
    for component in reversed([component for component in window.components if component.magic is not None]):
        layer = min(1 + constant.LAYERS_PER_STEP * (component.step - window.start), depth)
        while (component.magic, layer) in taken:
            layer -= 1
        if layer < 1:
            raise ValueError(f"depth {depth} cannot hold the magic states consumed at {component.magic}")
        taken.add((component.magic, layer))
        (neighbour,) = [tile for tile in program.steps[component.step].neighbours()[component.magic]
                        if tile in component.tiles]
        ports.append((component, layer, neighbour))
    return list(reversed(ports))


def _direction(src: Coord, dst: Coord) -> str:
    d_row, d_col = dst[0] - src[0], dst[1] - src[1]
    return {(0, 1): "+I", (0, -1): "-I", (1, 0): "+J", (-1, 0): "-J"}[(d_row, d_col)]


def spec(window: Window, program: Program, problem: Problem, depth: Optional[int] = None,
         qubit_z: str = QUBIT_Z, magic_z: str = MAGIC_Z) -> dict:
    # The lassynth specification of the window at depth layers, its own when depth is None.
    depth = layers(window) if depth is None else depth
    seats = _seats(window, program)
    ports = []
    for qubit in window.participants:
        cube_i, cube_j, _ = _cube(window, seats[qubit], 0)
        ports.append({"location": [cube_i, cube_j, 0], "direction": "+K", "z_basis_direction": qubit_z,
                      "function": f"in q{qubit}"})
    for qubit in window.participants:
        cube_i, cube_j, _ = _cube(window, seats[qubit], 0)
        ports.append({"location": [cube_i, cube_j, depth + 1], "direction": "-K", "z_basis_direction": qubit_z,
                      "function": f"out q{qubit}"})
    open_cubes = {tuple(port["location"]) for port in ports[:len(window.participants)]}
    for component, layer, neighbour in _magic_ports(window, program, depth):
        direction = _direction(component.magic, neighbour)
        across = "J" if direction[1] == "I" else "I"
        ports.append({"location": list(_cube(window, component.magic, layer)), "direction": direction,
                      "z_basis_direction": "K" if magic_z == "K" else across, "function": f"magic p{component.pair}"})
        open_cubes.add(_cube(window, component.magic, layer))

    # The window may not touch layer 0 outside the input ports, the column of any bystander or idle magic tile, or a
    # consumed magic tile away from its port.
    forbidden = set()
    for tile in window.box.tiles():
        forbidden.add(_cube(window, tile, 0))
    reserved = (list(window.bystanders.values()) + list(window.idle_magic)
                + [component.magic for component in window.components if component.magic])
    for tile in reserved:
        for layer in range(depth + 1):
            forbidden.add(_cube(window, tile, layer))
    forbidden -= open_cubes
    return {"max_i": window.box.width, "max_j": window.box.height, "max_k": depth + 1, "ports": ports,
            "stabilizers": stabilizers(window, problem),
            "optional": {"forbidden_cubes": [list(cube) for cube in sorted(forbidden)]}}


def pipes(window: Window, program: Program) -> dict[str, set]:
    # The tile program's pipes inside the window by axis, each as the cube it leaves, ports included.
    out = {"I": set(), "J": set(), "K": set()}

    def add(start, end):
        low, high = min(start, end), max(start, end)
        axis = "IJK"[next(i for i in range(3) if low[i] != high[i])]
        out[axis].add(low)

    depth = layers(window)
    seats = _seats(window, program)
    for qubit in window.participants:
        for layer in range(depth + 1):
            add(_cube(window, seats[qubit], layer), _cube(window, seats[qubit], layer + 1))

    for component in window.components:
        step = program.steps[component.step]
        lower = 1 + constant.LAYERS_PER_STEP * (component.step - window.start)
        upper = lower + 1
        path = _route_path(step, component.tiles, _z_party(step, component.tiles))
        if step.vertices[path[-1]].kind == MAGIC:
            for tile_a, tile_b in zip(path, path[1:]):
                add(_cube(window, tile_a, lower), _cube(window, tile_b, lower))
            continue
        turn = ancilla(path)
        for tile_a, tile_b in zip(path[:turn], path[1:turn + 1]):
            add(_cube(window, tile_a, lower), _cube(window, tile_b, lower))
        add(_cube(window, path[turn], lower), _cube(window, path[turn], upper))
        for tile_a, tile_b in zip(path[turn:], path[turn + 1:]):
            add(_cube(window, tile_a, upper), _cube(window, tile_b, upper))
    return out


def geometry(window: Window, program: Program) -> list[dict]:
    # Given values fixing every pipe to the tile program's and ruling out Y cubes, meaningful only at the window's own depth.
    mine = pipes(window, program)
    n_i, n_j, n_k = window.box.width, window.box.height, layers(window) + 1
    values = []
    for i in range(n_i):   # i, j, k index the cubes along the lassynth axes I, J and K
        for j in range(n_j):
            for k in range(n_k):
                for axis in "IJK":
                    values.append({"array": f"Exist{axis}", "indices": [i, j, k], "value": int((i, j, k) in mine[axis])})
                values.append({"array": "NodeY", "indices": [i, j, k], "value": 0})
    return values
