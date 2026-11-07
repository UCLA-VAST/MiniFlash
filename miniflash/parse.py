# Circuit ingestion. Reads OpenQASM 2 and normalizes it to the accepted Clifford+T gate set: h, cx, x, s,
# sdg, t, tdg, measure and barrier pass through, while z, y, cz, swap, id, rz at quarter turns and ccx are
# expanded in place. A gate outside this set raises ValueError, so nothing is silently dropped.
from __future__ import annotations

import math
from typing import Optional

from qiskit import qasm2
from qiskit.circuit import QuantumCircuit

ACCEPTED_GATES = ("h", "cx", "x", "s", "sdg", "t", "tdg", "measure", "barrier")

_EXPANSIONS = {
    "id": (),
    "z": (("h", (0,)), ("x", (0,)), ("h", (0,))),
    "y": (("h", (0,)), ("x", (0,)), ("h", (0,)), ("x", (0,))),
    "cz": (("h", (1,)), ("cx", (0, 1)), ("h", (1,))),
    "swap": (("cx", (0, 1)), ("cx", (1, 0)), ("cx", (0, 1))),
    "ccx": (("h", (2,)), ("cx", (1, 2)), ("tdg", (2,)), ("cx", (0, 2)), ("t", (2,)), ("cx", (1, 2)), ("tdg", (2,)), ("cx", (0, 2)), ("t", (1,)), ("t", (2,)), ("h", (2,)), ("cx", (0, 1)), ("t", (0,)), ("tdg", (1,)), ("cx", (0, 1))),
}

_RZ_QUARTER_TURNS = {
    0: (),
    1: (("s", (0,)),),
    2: (("h", (0,)), ("x", (0,)), ("h", (0,))),
    3: (("sdg", (0,)),),
}


def _rz_replacement(operation) -> Optional[tuple]:
    angle = float(operation.params[0])
    steps = round(angle / (math.pi / 2))
    if abs(angle - steps * math.pi / 2) > 1e-9:
        return None
    return _RZ_QUARTER_TURNS[steps % 4]


def _expand_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    if all(instruction.operation.name not in _EXPANSIONS and instruction.operation.name != "rz" for instruction in circuit.data):
        return circuit

    expanded = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for instruction in circuit.data:
        if instruction.operation.name == "rz":
            replacement = _rz_replacement(instruction.operation)
        else:
            replacement = _EXPANSIONS.get(instruction.operation.name)
        if replacement is None:
            expanded.append(instruction.operation, instruction.qubits, instruction.clbits)
        else:
            for name, positions in replacement:
                getattr(expanded, name)(*(instruction.qubits[position] for position in positions))
    return expanded


def _hoist_t(circuit: QuantumCircuit) -> QuantumCircuit:
    # Each t or tdg slides to its earliest commuting position on its qubit line, through s, sdg, t, tdg and
    # the cx control, flipping the dagger when it crosses an x and stopping at h, the cx target, measure and barrier.
    ops = [(instruction.operation.name, [circuit.find_bit(qubit).index for qubit in instruction.qubits], instruction)
           for instruction in circuit.data]
    result = []
    for name, qubits, instruction in ops:
        if name not in ("t", "tdg"):
            result.append([name, qubits, instruction])
            continue

        qubit = qubits[0]
        dagger = name == "tdg"
        insert_at = len(result)
        for position in range(len(result) - 1, -1, -1):
            other_name, other_qubits, _ = result[position]
            if qubit not in other_qubits:
                continue
            blocks = other_name in ("h", "measure", "barrier") or (other_name == "cx" and other_qubits[1] == qubit)
            if blocks:
                break
            if other_name == "x":
                dagger = not dagger
            insert_at = position
        result.insert(insert_at, ["tdg" if dagger else "t", [qubit], None])

    rebuilt = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    for name, qubits, instruction in result:
        if instruction is not None:
            rebuilt.append(instruction.operation, [rebuilt.qubits[qubit] for qubit in qubits],
                           instruction.clbits if hasattr(instruction, "clbits") else [])
        else:
            getattr(rebuilt, name)(qubits[0])
    return rebuilt


def parse(path) -> QuantumCircuit:
    circuit = _hoist_t(_expand_gates(qasm2.load(str(path))))
    for instruction in circuit.data:
        name = instruction.operation.name
        if name not in ACCEPTED_GATES:
            raise ValueError(f"unsupported construct '{name}' (accepted: h/cx/x/s/sdg/t/tdg/measure/barrier; z/y/cz/swap/id auto-expand)")
    return circuit
