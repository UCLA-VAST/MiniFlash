# Hierarchical Clifford+T benchmarks from Qualtran bloqs. A benchmark is `family-n`, a bloq family and its size,
# written as `<name>.json`, the hierarchy, and `<name>.qasm`, the same circuit flat for compilers that take no
# hierarchy. `decompose_bloq` gives a bloq's sub-bloqs and the wires between them, and every sub-bloq becomes a call
# of its own definition, built once and shared by all its callers, while bookkeeping bloqs only rename wires and leave
# no trace. A bloq Qualtran calls atomic is turned into gates through its cirq operation, so And is 4 T, Toffoli 7 T,
# and And's uncomputation is a measurement with a conditional CZ. TwoBitCSwap and SynthesizeLRCircuit have no cirq
# operation and get one here, cirq's CSWAP and the CNOTs of a Gauss-Jordan elimination. A definition of one item is
# spliced into its callers, so CNOT is the gate cx and not a call. A definition numbers its own qubits from 0, its
# inputs first, and a qubit Qualtran allocates inside takes the lowest free number, so freed workspace is reused as
# early as it can be and two sub-bloqs that could run side by side share workspace and wait for each other.
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Optional

import cirq
import numpy as np
from qualtran import Bloq, DecomposeNotImplementedError, DecomposeTypeError, LeftDangle, Register, Soquet
from qualtran.bloqs.basic_gates import TwoBitCSwap
from qualtran.bloqs.basic_gates.x_basis import _XVector
from qualtran.bloqs.basic_gates.z_basis import _ZVector
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.bloqs.gf_arithmetic.gf2_multiplication import SynthesizeLRCircuit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miniflash.hierarchy import GATES, Def, Hierarchy, Item  # noqa: E402

Quregs = dict[str, np.ndarray]   # local qubits by register name, shape reg.shape + (reg.bitsize,)
QUARTER = {0: (), 1: ("t",), 2: ("s",), 3: ("s", "t"), 4: ("s", "s"), 5: ("s", "s", "t"), 6: ("sdg",), 7: ("tdg",)}


# The local qubits of one definition, the lowest free index first.
class Pool(cirq.QubitManager):
    def __init__(self):
        self.size = 0
        self.free = set()

    def qalloc(self, count: int, dim: int = 2) -> list[cirq.LineQubit]:
        out = []
        for _ in range(count):
            index = min(self.free) if self.free else self.size
            self.free.discard(index)
            self.size = max(self.size, index + 1)
            out.append(cirq.LineQubit(index))
        return out

    def qborrow(self, count: int, dim: int = 2) -> list[cirq.LineQubit]:
        return self.qalloc(count, dim)

    def qfree(self, qubits) -> None:
        self.free.update(qubit.x for qubit in qubits)


def translate(op: cirq.Operation) -> Optional[list[tuple[str, tuple]]]:
    # None when the operation is not a gate of the benchmark set, and global phases are dropped.
    gate = op.gate
    qubits = op.qubits
    if isinstance(gate, (cirq.IdentityGate, cirq.GlobalPhaseGate)):
        return []
    if isinstance(gate, cirq.MeasurementGate) and len(qubits) == 1:
        return [("measure", qubits)]
    if isinstance(gate, cirq.ResetChannel):
        return [("reset", qubits)]
    if isinstance(gate, cirq.HPowGate) and gate.exponent == 1:
        return [("h", qubits)]
    if isinstance(gate, cirq.CXPowGate) and gate.exponent == 1:
        return [("cx", qubits)]
    if isinstance(gate, cirq.CZPowGate) and gate.exponent == 1:
        return [("h", qubits[1:]), ("cx", qubits), ("h", qubits[1:])]
    if isinstance(gate, (cirq.ZPowGate, cirq.XPowGate, cirq.YPowGate)) and not cirq.is_parameterized(gate):
        quarters = float(gate.exponent) * 4
        if abs(quarters - round(quarters)) > 1e-9:
            return None
        z_gates = [(name, qubits) for name in QUARTER[round(quarters) % 8]]
        if isinstance(gate, cirq.ZPowGate):
            return z_gates
        x_gates = [("x", qubits)] if round(quarters) % 8 == 4 else [("h", qubits)] + z_gates + [("h", qubits)]
        return x_gates if isinstance(gate, cirq.XPowGate) else [("sdg", qubits)] + x_gates + [("s", qubits)]
    return None


def _bare(op: cirq.Operation) -> cirq.Operation:
    return op.without_classical_controls() if isinstance(op, cirq.ClassicallyControlledOperation) else op


def gates_of(op: cirq.Operation, pool: Pool) -> list[Item]:
    # A classically controlled gate is conditioned on the qubit its measurement key was read from.
    out: list[Item] = []
    measured: dict[str, int] = {}
    keep = lambda candidate: translate(_bare(candidate)) is not None
    for sub_op in cirq.decompose(op, keep=keep, context=cirq.DecompositionContext(pool)):
        cond = None
        if isinstance(sub_op, cirq.ClassicallyControlledOperation):
            (control,) = sub_op.classical_controls
            (key,) = control.keys
            cond = measured[key.name]
        for name, qubits in translate(_bare(sub_op)):
            if name == "measure":
                measured[sub_op.gate.key] = qubits[0].x
            out.append((name, tuple(qubit.x for qubit in qubits), cond))
    return out


def _effect(bloq: Bloq) -> Optional[tuple[str, ...]]:
    # The gates returning the qubit an effect bloq discards to |0>, None for any other bloq.
    if isinstance(bloq, _ZVector) and not bloq.state:
        return ("x",) if bloq.bit else ()
    if isinstance(bloq, _XVector) and not bloq.state:
        return ("h", "x") if bloq.bit else ("h",)
    return None


def linear(matrix) -> list[tuple[int, int]]:
    # The CNOTs (control, target) taking q to matrix @ q over GF(2) are the row operations of a Gauss-Jordan
    # elimination, undone last to first.
    reduced = np.array(matrix, dtype=int) % 2
    cnots = []
    for col in range(len(reduced)):
        if not reduced[col, col]:
            pivot = next(row for row in range(col + 1, len(reduced)) if reduced[row, col])
            reduced[col] ^= reduced[pivot]
            cnots.append((pivot, col))
        for row in range(len(reduced)):
            if row != col and reduced[row, col]:
                reduced[row] ^= reduced[col]
                cnots.append((col, row))
    return cnots[::-1]


def _labels(reg: Register) -> list[str]:
    return ["_".join([reg.name, *map(str, idx), *([str(bit)] if reg.bitsize > 1 else [])])
            for idx in reg.all_idxs() for bit in range(reg.bitsize)]


def _gather(reg: Register, soquet, wires: dict[Soquet, np.ndarray]) -> np.ndarray:
    out = np.empty(reg.shape + (reg.bitsize,), dtype=object)
    for idx in reg.all_idxs():
        out[idx] = wires[np.asarray(soquet)[idx]]
    return out


def _qubits(quregs: Quregs) -> set:
    return {qubit for qureg in quregs.values() for qubit in np.asarray(qureg).flatten()}


# The definitions of bloqs, each built once.
class Builder:
    def __init__(self):
        self.defs: dict[Bloq, Def] = {}
        self.names = set(GATES)

    def name(self, bloq: Bloq) -> str:
        base = re.sub(r"[^0-9a-z]+", "_", str(bloq).replace("†", " dag").lower()).strip("_") or "bloq"
        if not base[0].isalpha():
            base = "b_" + base

        name = base
        suffix = 1
        while name in self.names:
            suffix += 1
            name = f"{base}_{suffix}"
        self.names.add(name)
        return name

    def define(self, bloq: Bloq) -> Def:
        if bloq in self.defs:
            return self.defs[bloq]

        pool = Pool()
        quregs = {reg.name: np.array(pool.qalloc(reg.total_bits()), dtype=object).reshape(reg.shape + (reg.bitsize,))
                  for reg in bloq.signature.lefts()}
        ins = tuple(range(pool.size))
        body: list[Item] = []
        try:
            cbloq = bloq.decompose_bloq()
        except (DecomposeTypeError, DecomposeNotImplementedError):
            outs = self.apply(bloq, quregs, pool, body)
        else:
            outs = self.composite(cbloq, quregs, pool, body)
        pool.qfree(_qubits(quregs) - _qubits(outs))
        out_ids = tuple(qubit.x for reg in bloq.signature.rights() for qubit in np.asarray(outs[reg.name]).flatten())
        dirty = set(range(pool.size)) - pool.free - set(out_ids)
        if dirty:
            raise ValueError(f"{bloq}: local qubits {sorted(dirty)} are neither freed nor returned")

        labels: list[Optional[str]] = [None] * pool.size
        for regs, ids in ((bloq.signature.lefts(), ins), (bloq.signature.rights(), out_ids)):
            for label, index in zip([label for reg in regs for label in _labels(reg)], ids):
                if labels[index] is None:
                    while label in labels:
                        label += "_"
                    labels[index] = label
        for index in range(pool.size):
            if labels[index] is None:
                label = f"anc_{index}"
                while label in labels:
                    label += "_"
                labels[index] = label

        self.defs[bloq] = Def(self.name(bloq), " ".join(repr(bloq).split())[:200], tuple(labels), ins, out_ids, tuple(body))
        return self.defs[bloq]

    def apply(self, bloq: Bloq, quregs: Quregs, pool: Pool, body: list[Item]) -> Quregs:
        # Appends the gates of an atomic or bookkeeping bloq to body and gives its output registers.
        effect = _effect(bloq)
        if effect is not None:
            (qubit,) = _qubits(quregs)
            body += [(gate, (qubit.x,), None) for gate in effect]
            return {}
        if isinstance(bloq, SynthesizeLRCircuit):
            qubits = quregs["q"].flatten()
            cnots = linear(bloq.matrix)
            body += [("cx", (qubits[control].x, qubits[target].x), None)
                     for control, target in (cnots[::-1] if bloq.is_adjoint else cnots)]
            return quregs

        if isinstance(bloq, TwoBitCSwap):
            op = cirq.CSWAP(*(qubit for name in ("ctrl", "x", "y") for qubit in quregs[name].flatten()))
            outs = quregs
        else:
            op, outs = bloq.as_cirq_op(pool, **quregs)
        if op is not None:
            try:
                body += gates_of(op, pool)
            except ValueError as error:
                raise ValueError(f"{bloq!r} is not Clifford+T: {str(error)[:200]}") from None
        return outs

    def call(self, bloq: Bloq, quregs: Quregs, pool: Pool, body: list[Item]) -> Quregs:
        # Free qubits are lent for the rest of the definition's qubits, and a definition of one item is spliced in.
        definition = self.define(bloq)
        args = [qubit for reg in bloq.signature.lefts() for qubit in quregs[reg.name].flatten()]
        args += pool.qalloc(len(definition.qubits) - len(args))
        ids = tuple(qubit.x for qubit in args)
        if len(definition.body) <= 1:
            body += [(gate, tuple(ids[i] for i in local_qubits), None if cond is None else ids[cond])
                     for gate, local_qubits, cond in definition.body]
        else:
            body.append((definition.name, ids, None))

        outs = {}
        offset = 0
        for reg in bloq.signature.rights():
            n_bits = reg.total_bits()
            outs[reg.name] = np.array([args[i] for i in definition.outs[offset:offset + n_bits]],
                                      dtype=object).reshape(reg.shape + (reg.bitsize,))
            offset += n_bits
        pool.qfree(set(args) - _qubits(outs))
        return outs

    def composite(self, cbloq, quregs: Quregs, pool: Pool, body: list[Item]) -> Quregs:
        # Walks a decomposition in topological order, the wires carrying local qubits.
        wires = {Soquet(LeftDangle, reg, idx): quregs[reg.name][idx]
                 for reg in cbloq.signature.lefts() for idx in reg.all_idxs()}
        for binst, in_soqs, out_soqs in cbloq.iter_bloqsoqs():
            bloq = binst.bloq
            given = {reg.name: _gather(reg, in_soqs[reg.name], wires) for reg in bloq.signature.lefts()}
            if isinstance(bloq, _BookkeepingBloq):
                outs = self.apply(bloq, given, pool, body)
                pool.qfree(_qubits(given) - _qubits(outs))
            else:
                outs = self.call(bloq, given, pool, body)
            for reg, soquet in zip(bloq.signature.rights(), out_soqs):
                for idx in reg.all_idxs():
                    wires[np.asarray(soquet)[idx]] = np.asarray(outs[reg.name])[idx]

        final = cbloq.final_soqs()
        return {reg.name: _gather(reg, final[reg.name], wires) for reg in cbloq.signature.rights()}


def hierarchy(bloq: Bloq, source: str = "") -> Hierarchy:
    # The definition of bloq and every definition it reaches.
    builder = Builder()
    root = builder.define(bloq)
    by_name = {definition.name: definition for definition in builder.defs.values()}

    reached = {root.name}
    stack = [root.name]
    while stack:
        for gate, _, _ in by_name[stack.pop()].body:
            if gate in by_name and gate not in reached:
                reached.add(gate)
                stack.append(gate)
    return Hierarchy([definition for definition in builder.defs.values() if definition.name in reached], root.name, source)


def _prime(n_bits: int) -> int:
    # The largest prime below 2**n_bits.
    candidate = 2 ** n_bits - 1
    while any(candidate % divisor == 0 for divisor in range(2, int(candidate ** 0.5) + 1)):
        candidate -= 1
    return candidate


def _add(size: int) -> Bloq:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    return Add(QUInt(size))


def _modadd(size: int) -> Bloq:
    from qualtran.bloqs.mod_arithmetic import ModAdd
    return ModAdd(size, _prime(size))


def _modmul(size: int) -> Bloq:
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic import CModMulK
    return CModMulK(QUInt(size), k=3, mod=_prime(size))


def _modexp(size: int) -> Bloq:
    from qualtran.bloqs.cryptography.rsa import ModExp
    return ModExp(base=3, mod=_prime(size), exp_bitsize=size, x_bitsize=size)


def _gf2mul(size: int) -> Bloq:
    from qualtran.bloqs.gf_arithmetic import GF2Multiplication
    return GF2Multiplication(size)


def _gf2inv(size: int) -> Bloq:
    from qualtran.bloqs.gf_arithmetic import GF2Inverse
    return GF2Inverse(size)


def _multiand(size: int) -> Bloq:
    from qualtran.bloqs.mcmt import MultiAnd
    return MultiAnd((1,) * size)


def _lte(size: int) -> Bloq:
    from qualtran.bloqs.arithmetic import LessThanEqual
    return LessThanEqual(size, size)


def _qrom(size: int) -> Bloq:
    from qualtran.bloqs.data_loading.qrom import QROM
    data = np.random.default_rng(0).integers(0, 2 ** size, size=2 ** size)
    return QROM.build_from_data(data)


def _hubbard(size: int) -> Bloq:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import SelectHubbard
    return SelectHubbard(size, size)


FAMILIES: dict[str, tuple[Callable[[int], Bloq], str]] = {
    "add": (_add, "n-bit adder, a + b into b (Gidney)"),
    "modadd": (_modadd, "n-bit modular adder, modulus the largest prime below 2^n"),
    "modmul": (_modmul, "controlled n-bit modular multiplication by 3, same modulus"),
    "modexp": (_modexp, "n-bit modular exponentiation 3^e, n-bit exponent, same modulus: n controlled multiplications"),
    "gf2mul": (_gf2mul, "GF(2^n) multiplication, out of place"),
    "gf2inv": (_gf2inv, "GF(2^n) inversion by Itoh-Tsujii: squarings and multiplications"),
    "multiand": (_multiand, "AND of n controls with its ladder of ancillas"),
    "lte": (_lte, "n-bit comparison a <= b into one qubit"),
    "qrom": (_qrom, "table lookup: 2^n entries of n bits, seeded random data"),
    "hubbard": (_hubbard, "SELECT of the n x n Hubbard model (qubitization)"),
}


def build(name: str) -> Hierarchy:
    # The benchmark `family-n`.
    family, _, size = name.rpartition("-")
    bloq = FAMILIES[family][0](int(size))
    from qualtran._version import __version__
    return hierarchy(bloq, f"{name}: {' '.join(repr(bloq).split())[:120]}, qualtran {__version__}")


def describe(benchmark: Hierarchy) -> str:
    # One line per definition with how often it appears, its qubits, and its T and CX gates.
    seen = benchmark.instances()
    lines = [f"{'definition':<24}{'calls':>7}{'in':>5}{'out':>5}{'work':>6}{'t':>8}{'cx':>8}  calls"]
    for definition in reversed(list(benchmark.defs.values())):
        counts = benchmark.counts(definition.name)
        callees = ", ".join(f"{callee} x{times}" for callee, times in benchmark.calls(definition.name).items())
        lines.append(f"{definition.name:<24}{seen.get(definition.name, 0):>7}{len(definition.ins):>5}"
                     f"{len(definition.outs):>5}{len(definition.workspace):>6}"
                     f"{counts.get('t', 0) + counts.get('tdg', 0):>8}{counts.get('cx', 0):>8}  {callees}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="hierarchical Clifford+T benchmarks from Qualtran bloqs")
    parser.add_argument("names", nargs="*", help="benchmarks, each family-n")
    parser.add_argument("--list", action="store_true", help="list the families")
    parser.add_argument("-o", "--out", default="benchmarks/qualtran", help="directory for the .json and .qasm files")
    args = parser.parse_args()
    if args.list or not args.names:
        for family, (_, text) in FAMILIES.items():
            print(f"{family:<10}{text}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        benchmark = build(name)
        benchmark.save(out_dir / f"{name}.json")
        (out_dir / f"{name}.qasm").write_text(benchmark.to_qasm())
        print(f"{name}: {benchmark.n_qubits} qubits, {len(benchmark.problem.pairs)} pairs, depth {benchmark.problem.depth}, "
              f"{len(benchmark.defs)} definitions, {sum(benchmark.instances().values())} calls")
        print(describe(benchmark))


if __name__ == "__main__":
    main()
