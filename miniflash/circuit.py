# Input side: a Clifford+T circuit becomes a Problem, a DAG of ordered pairs. A pair is (z, x), the party
# attached through its Z boundary at the top or bottom and the party attached through its X boundary at the
# left or right, so a CX is (control, target) and a T gate is (qubit, MAGIC) with the magic state as the X
# party. Cliffords are free in the execution model, so the reader drops them but counts what it dropped.
# Dependencies are program order per qubit, and MAGIC never induces one because every magic state is fresh.
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

MAGIC = -1
CLIFFORD = {"h", "x", "y", "z", "s", "sdg", "sx", "sxdg", "id"}


@dataclass(frozen=True)
class Pair:
    z: int
    x: int
    tag: str   # cx, t or tdg

    @property
    def is_magic(self) -> bool:
        return self.x == MAGIC

    @property
    def qubits(self) -> tuple[int, ...]:
        return (self.z,) if self.is_magic else (self.z, self.x)

    def __str__(self) -> str:
        return f"{self.tag} q{self.z}" if self.is_magic else f"{self.tag} (q{self.z}, q{self.x})"


# preds[i] lists the pairs that must run before pair i.
@dataclass(frozen=True)
class Problem:
    n_qubits: int
    pairs: tuple[Pair, ...]
    preds: tuple[tuple[int, ...], ...]
    dropped: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_pairs(cls, n_qubits: int, pairs: Sequence[Pair], dropped: Optional[dict[str, int]] = None) -> Problem:
        # Each pair waits for the last earlier pair on each of its qubits.
        last: dict[int, int] = {}
        preds = []
        for pair_id, pair in enumerate(pairs):
            preds.append(tuple(sorted({last[qubit] for qubit in pair.qubits if qubit in last})))
            for qubit in pair.qubits:
                last[qubit] = pair_id
        return cls(n_qubits, tuple(pairs), tuple(preds), dict(dropped or {}))

    @classmethod
    def builder(cls, n_qubits: int) -> Builder:
        return Builder(n_qubits)

    @property
    def qubits(self) -> range:
        return range(self.n_qubits)

    def succs(self) -> tuple[tuple[int, ...], ...]:
        out: list[list[int]] = [[] for _ in self.pairs]
        for pair_id, predecessors in enumerate(self.preds):
            for predecessor in predecessors:
                out[predecessor].append(pair_id)
        return tuple(tuple(successors) for successors in out)

    def layers(self) -> list[list[int]]:
        # ASAP layers, pair i one layer below its deepest predecessor.
        level = [0] * len(self.pairs)
        for pair_id, predecessors in enumerate(self.preds):
            level[pair_id] = max((level[predecessor] + 1 for predecessor in predecessors), default=0)

        out: list[list[int]] = [[] for _ in range(max(level, default=-1) + 1)]
        for pair_id, layer in enumerate(level):
            out[layer].append(pair_id)
        return out

    @property
    def depth(self) -> int:
        return len(self.layers())

    def interaction(self) -> dict[tuple[int, int], int]:
        # CX pairs joining (a, b) with a < b, since T pairs touch a single qubit and add no edge.
        weights: dict[tuple[int, int], int] = {}
        for pair in self.pairs:
            if not pair.is_magic:
                edge = (min(pair.z, pair.x), max(pair.z, pair.x))
                weights[edge] = weights.get(edge, 0) + 1
        return weights

    def criticality(self) -> list[int]:
        # Length of the longest dependency chain starting at each pair, itself included.
        crit = [1] * len(self.pairs)
        for pair_id in range(len(self.pairs) - 1, -1, -1):
            for predecessor in self.preds[pair_id]:
                crit[predecessor] = max(crit[predecessor], crit[pair_id] + 1)
        return crit


# Collects pairs in program order and build derives the DAG.
class Builder:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.pairs: list[Pair] = []

    def add_cx(self, control: int, target: int) -> Builder:
        self.pairs.append(Pair(control, target, "cx"))
        return self

    def add_t(self, qubit: int) -> Builder:
        self.pairs.append(Pair(qubit, MAGIC, "t"))
        return self

    def add_tdg(self, qubit: int) -> Builder:
        self.pairs.append(Pair(qubit, MAGIC, "tdg"))
        return self

    def build(self) -> Problem:
        return Problem.from_pairs(self.n_qubits, self.pairs)


def from_circuit(circuit) -> Problem:
    # The circuit is a normalized Clifford+T QuantumCircuit as parse returns it.
    pairs: list[Pair] = []
    dropped: Counter = Counter()
    for instruction in circuit.data:
        name = instruction.operation.name
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if name == "cx":
            pairs.append(Pair(qubits[0], qubits[1], "cx"))
        elif name in ("t", "tdg"):
            pairs.append(Pair(qubits[0], MAGIC, name))
        elif name in CLIFFORD or name in ("measure", "barrier"):
            dropped[name] += 1
        else:
            raise ValueError(f"unsupported gate {name}")
    return Problem.from_pairs(circuit.num_qubits, pairs, dict(dropped))


def read_qasm(path) -> Problem:
    from .parse import parse
    return from_circuit(parse(path))


def to_dot(problem: Problem, path=None) -> str:
    # Ranks follow the ASAP layers and every edge carries the shared qubit.
    lines = ["digraph problem {", "  rankdir=TB; node [shape=box, style=filled, fontname=Helvetica];"]
    for layer in problem.layers():
        names = " ".join(f"p{pair_id}" for pair_id in layer)
        lines.append(f"  {{ rank=same; {names}; }}")
    for pair_id, pair in enumerate(problem.pairs):
        fill = "#add2c2" if pair.is_magic else "#c9bdd3"
        lines.append(f'  p{pair_id} [label="{pair_id}: {pair}", fillcolor="{fill}"];')
    for pair_id, predecessors in enumerate(problem.preds):
        for predecessor in predecessors:
            shared = sorted(set(problem.pairs[pair_id].qubits) & set(problem.pairs[predecessor].qubits))
            label = ",".join(f"q{qubit}" for qubit in shared)
            lines.append(f'  p{predecessor} -> p{pair_id} [label="{label}"];')
    lines.append("}")

    source = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).write_text(source)
    return source
