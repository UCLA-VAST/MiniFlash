# Hierarchy: a Clifford+T circuit together with the call tree it was built from. A benchmark is a set of
# definitions, one per subroutine, each naming its local qubits and listing its body in program order as
# gates on local qubits and calls of other definitions. The local qubits alive at entry are its inputs and
# those alive at exit its outputs, and the rest is workspace the body takes clean and returns clean, so a
# caller may lend any free qubit for it. Flattening the root gives the pair DAG in program order and one Node
# per call spanning a contiguous range of pairs, so every cut through the tree is a partition of the problem.
# A gate may carry a condition, the qubit whose last measurement decides whether it runs, and the problem
# keeps such a gate as if it always ran, the worst case a static schedule has to provide for.
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .circuit import MAGIC, Pair, Problem

GATES = ("h", "x", "s", "sdg", "t", "tdg", "cx", "measure", "reset")
Item = tuple[str, tuple[int, ...], Optional[int]]   # (gate or definition, qubits, condition)


# ins and outs index qubits, and bloq says where the subroutine came from.
@dataclass(frozen=True)
class Def:
    name: str
    bloq: str
    qubits: tuple[str, ...]
    ins: tuple[int, ...]
    outs: tuple[int, ...]
    body: tuple[Item, ...]

    @property
    def workspace(self) -> tuple[int, ...]:
        pins = set(self.ins) | set(self.outs)
        return tuple(qubit for qubit in range(len(self.qubits)) if qubit not in pins)

    def preds(self) -> tuple[tuple[int, ...], ...]:
        # Each item waits for the last earlier item on each of its qubits and for the measurement it is conditioned on.
        last: dict[int, int] = {}
        out = []
        for item_index, (_, qubits, cond) in enumerate(self.body):
            used = qubits if cond is None else qubits + (cond,)
            out.append(tuple(sorted({last[qubit] for qubit in used if qubit in last})))
            for qubit in qubits:
                last[qubit] = item_index
        return tuple(out)

    def to_dict(self) -> dict:
        return {"name": self.name, "bloq": self.bloq, "qubits": list(self.qubits), "in": list(self.ins), "out": list(self.outs),
                "body": [[gate, list(qubits)] if cond is None else [gate, list(qubits), cond] for gate, qubits, cond in self.body]}

    @classmethod
    def from_dict(cls, data: dict) -> Def:
        body = tuple((item[0], tuple(item[1]), item[2] if len(item) > 2 else None) for item in data["body"])
        return cls(data["name"], data["bloq"], tuple(data["qubits"]), tuple(data["in"]), tuple(data["out"]), body)


# One call in the flat circuit, spanning pairs start..stop-1 of the problem with its children's included.
@dataclass
class Node:
    name: str
    qubits: tuple[int, ...]   # in the order of the definition's
    start: int
    stop: int = 0
    children: list[Node] = field(default_factory=list)

    @property
    def pairs(self) -> range:
        return range(self.start, self.stop)

    def walk(self) -> Iterator[Node]:
        # This node and everything below it, parents first.
        yield self
        for child in self.children:
            yield from child.walk()


class Hierarchy:
    def __init__(self, defs: Iterable[Def], root: str, source: str = ""):
        # The root's local qubits are the circuit's, and source is free text on where the benchmark came from.
        self.defs: dict[str, Def] = {definition.name: definition for definition in defs}
        self.root = root
        self.source = source

        self.gates: list[Item] = []
        pairs: list[Pair] = []
        dropped: Counter = Counter()
        self.tree = self._flatten(root, tuple(range(len(self.defs[root].qubits))), pairs, dropped)
        self.problem = Problem.from_pairs(len(self.defs[root].qubits), pairs, dict(dropped))

    def _flatten(self, name: str, args: tuple[int, ...], pairs: list[Pair], dropped: Counter) -> Node:
        node = Node(name, args, len(pairs))
        for gate, local_qubits, cond in self.defs[name].body:
            qubits = tuple(args[local] for local in local_qubits)
            if gate in self.defs:
                node.children.append(self._flatten(gate, qubits, pairs, dropped))
                continue
            self.gates.append((gate, qubits, None if cond is None else args[cond]))
            if gate == "cx":
                pairs.append(Pair(qubits[0], qubits[1], "cx"))
            elif gate in ("t", "tdg"):
                pairs.append(Pair(qubits[0], MAGIC, gate))
            else:
                dropped[gate] += 1
        node.stop = len(pairs)
        return node

    @property
    def n_qubits(self) -> int:
        return self.problem.n_qubits

    def calls(self, name: str) -> dict[str, int]:
        # How often each definition is called from the body of name.
        return dict(Counter(gate for gate, _, _ in self.defs[name].body if gate in self.defs))

    def counts(self, name: str) -> dict[str, int]:
        # Gates of one call of name, everything it calls included.
        memo: dict[str, Counter] = {}

        def of(definition: str) -> Counter:
            if definition not in memo:
                total: Counter = Counter()
                for gate, _, _ in self.defs[definition].body:
                    total.update(of(gate) if gate in self.defs else {gate: 1})
                memo[definition] = total
            return memo[definition]
        return dict(of(name))

    def instances(self) -> dict[str, int]:
        # How many times each definition appears in the flat circuit.
        return dict(Counter(node.name for node in self.tree.walk()))

    def to_qasm(self) -> str:
        # Conditional gates become unconditional and measure and reset are left out.
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";']
        if self.source:
            lines.append(f"// {self.source}")
        lines.append(f"qreg q[{self.n_qubits}];")
        for gate, qubits, _ in self.gates:
            if gate not in ("measure", "reset"):
                lines.append(f"{gate} {','.join(f'q[{qubit}]' for qubit in qubits)};")
        return "\n".join(lines) + "\n"

    def save(self, path) -> None:
        body = {"source": self.source, "root": self.root, "defs": [definition.to_dict() for definition in self.defs.values()]}
        Path(path).write_text(json.dumps(body, separators=(",", ":")) + "\n")

    @classmethod
    def load(cls, path) -> Hierarchy:
        body = json.loads(Path(path).read_text())
        return cls([Def.from_dict(entry) for entry in body["defs"]], body["root"], body.get("source", ""))
