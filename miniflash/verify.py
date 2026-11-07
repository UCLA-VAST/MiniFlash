# Independent checker: does a Program execute a Problem under the execution model? It reads only the step
# graphs, never the solver's intermediate objects, so it also checks programs produced by adapters for other
# compilers. Per step every qubit sits on exactly one tile, edges join 4-neighbours, and every component with
# an edge realises exactly one pair, whose Z party joins through its Z side and X party through its X side
# with no other qubit or foreign magic tile inside. Across steps each pair runs exactly once after all its
# predecessors, a qubit moves at most one tile per axis per step, and a step where anything moves has no edges.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .circuit import Problem
from .factory import Factory, supply_limits
from .program import IDLE, MAGIC, QUBIT, ROUTE, Program, Step


class VerifyError(AssertionError):
    pass


@dataclass(frozen=True)
class Report:
    steps: int
    walk_steps: int
    height: int
    width: int
    when: dict[int, int]   # the step each pair executed at

    @property
    def volume(self) -> int:
        return self.steps * self.height * self.width


def _adjacent(tile_a: tuple[int, int], tile_b: tuple[int, int]) -> bool:
    return abs(tile_a[0] - tile_b[0]) + abs(tile_a[1] - tile_b[1]) == 1


def _check_step(step_index: int, step: Step, problem: Problem, program: Program) -> dict[int, int]:
    # Gives {pair: 1} for the pairs executed at this step.
    for vertex in step.vertices.values():
        if not (0 <= vertex.tile[0] < program.height and 0 <= vertex.tile[1] < program.width):
            raise VerifyError(f"step {step_index}: vertex {vertex.tile} off the grid")
    seats = step.qubit_tiles()
    qubit_vertices = step.of_kind(QUBIT)
    if len(qubit_vertices) != problem.n_qubits or set(seats) != set(problem.qubits):
        raise VerifyError(f"step {step_index}: expected every qubit exactly once, "
                          f"found {sorted(vertex.id for vertex in qubit_vertices)}")
    for edge in step.edges:
        tile_a, tile_b = tuple(edge)
        if not _adjacent(tile_a, tile_b):
            raise VerifyError(f"step {step_index}: edge {tile_a}-{tile_b} joins non-adjacent tiles")

    done: dict[int, int] = {}
    for component in step.components():
        vertices = [step.vertices[tile] for tile in component]
        ids = ({vertex.id for vertex in vertices if vertex.kind == ROUTE}
               | {vertex.id for vertex in vertices if vertex.kind == MAGIC and vertex.id != IDLE})
        if len(ids) != 1:
            raise VerifyError(f"step {step_index}: component {sorted(component)} carries pair ids {sorted(ids)}")
        (pair_id,) = ids
        if not 0 <= pair_id < len(problem.pairs):
            raise VerifyError(f"step {step_index}: unknown pair {pair_id}")
        if pair_id in done:
            raise VerifyError(f"step {step_index}: pair {pair_id} realised twice")
        pair = problem.pairs[pair_id]
        if any(vertex.kind == MAGIC and vertex.id == IDLE for vertex in vertices):
            raise VerifyError(f"step {step_index}: pair {pair_id} merges an idle magic tile")
        qubit_tiles = {vertex.id: vertex.tile for vertex in vertices if vertex.kind == QUBIT}
        if set(qubit_tiles) != set(pair.qubits):
            raise VerifyError(f"step {step_index}: pair {pair_id} {pair} joins qubits {sorted(qubit_tiles)}")

        neighbours = step.neighbours()
        z_tile = qubit_tiles[pair.z]
        for neighbour in neighbours[z_tile]:
            if neighbour[1] != z_tile[1]:
                raise VerifyError(f"step {step_index}: pair {pair_id} leaves q{pair.z} through the wrong side {z_tile}-{neighbour}")
        if pair.is_magic:
            magics = [vertex for vertex in vertices if vertex.kind == MAGIC]
            if len(magics) != 1:
                raise VerifyError(f"step {step_index}: pair {pair_id} consumes {len(magics)} magic tiles")
            x_tile = magics[0].tile
        else:
            x_tile = qubit_tiles[pair.x]
            for neighbour in neighbours[x_tile]:
                if neighbour[0] != x_tile[0]:
                    raise VerifyError(f"step {step_index}: pair {pair_id} enters q{pair.x} through the wrong side {x_tile}-{neighbour}")
        if not neighbours[z_tile] or not neighbours[x_tile]:
            raise VerifyError(f"step {step_index}: pair {pair_id} does not reach both parties")
        done[pair_id] = 1

    # Route vertices and consumed magic must belong to some component.
    for vertex in step.vertices.values():
        if vertex.kind == ROUTE and vertex.id not in done:
            raise VerifyError(f"step {step_index}: route tile {vertex.tile} of pair {vertex.id} is disconnected")
        if vertex.kind == MAGIC and vertex.id != IDLE and vertex.id not in done:
            raise VerifyError(f"step {step_index}: magic {vertex.tile} consumed by pair {vertex.id} without a route")
    return done


def _check_move(step_index: int, before: Step, after: Step) -> bool:
    # True when some qubit moved between the two steps.
    tiles_before = before.qubit_tiles()
    tiles_after = after.qubit_tiles()
    moved = False
    for qubit, tile in tiles_after.items():
        distance = max(abs(tile[0] - tiles_before[qubit][0]), abs(tile[1] - tiles_before[qubit][1]))
        if distance > 1:
            raise VerifyError(f"step {step_index}: q{qubit} jumps from {tiles_before[qubit]} to {tile}")
        moved |= distance == 1
    if moved and not after.is_walk:
        raise VerifyError(f"step {step_index}: qubits move and gates execute in the same step")
    return moved


def _check_cooldown(step_index: int, step: Step, last: dict[tuple[int, int], int], cooldown: int, walked: bool) -> None:
    # A magic tile consumed at step s may not be consumed again before step s + cooldown.
    # A stage after a walk re-provisions its delivery tiles at new coordinates, so the record restarts after every walk.
    if walked:
        last.clear()
        return
    for vertex in step.of_kind(MAGIC):
        if vertex.id == IDLE:
            continue
        if vertex.tile in last and step_index - last[vertex.tile] < cooldown:
            raise VerifyError(f"step {step_index}: magic {vertex.tile} consumed by pair {vertex.id} only "
                              f"{step_index - last[vertex.tile]} steps after step {last[vertex.tile]}, cooldown {cooldown}")
        last[vertex.tile] = step_index


def verify(problem: Problem, program: Program, factory: Optional[Factory] = None, factories: int = 0) -> Report:
    # With a factory alone no delivery tile is consumed again within factory.period steps, and with F shared
    # factories the k-th state consumed comes no earlier than routed step k x period / F.
    magic_cooldown, factories, period = supply_limits(factory, factories)
    when: dict[int, int] = {}
    walks = 0
    last_magic: dict[tuple[int, int], int] = {}
    states = 0
    for step_index, step in enumerate(program.steps):
        walked = step_index > 0 and _check_move(step_index, program.steps[step_index - 1], step)
        walks += walked
        if magic_cooldown > 1:
            _check_cooldown(step_index, step, last_magic, magic_cooldown, walked)
        if factories:
            states += sum(1 for vertex in step.of_kind(MAGIC) if vertex.id != IDLE)
            if states > ((step_index - walks) * factories) // period:
                raise VerifyError(f"step {step_index}: state {states} consumed before {factories} factories at one per {period} steps "
                                  f"could produce it (routed step {step_index - walks})")

        for pair_id in _check_step(step_index, step, problem, program):
            if pair_id in when:
                raise VerifyError(f"pair {pair_id} executed at steps {when[pair_id]} and {step_index}")
            for predecessor in problem.preds[pair_id]:
                if predecessor not in when or when[predecessor] >= step_index:
                    raise VerifyError(f"step {step_index}: pair {pair_id} runs before its predecessor {predecessor}")
            when[pair_id] = step_index

    missing = [pair_id for pair_id in range(len(problem.pairs)) if pair_id not in when]
    if missing:
        raise VerifyError(f"{len(missing)} pairs never executed, first {missing[:5]}")
    return Report(len(program.steps), walks, program.height, program.width, when)
