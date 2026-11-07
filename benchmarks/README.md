# Benchmark circuits

206 QASM circuits organized by family.

| folder | count | description |
|---|---|---|
| `algorithms/` | 41 | GHZ (3 to 16 qubits), Bernstein-Vazirani and Deutsch-Jozsa (16 to 100), QFT, QPE, QAOA, Grover, Shor-15, HWB, Hamming-15, VQE, W state |
| `arithmetic/` | 17 | ripple-carry, VBE, QCLA, carry-select and carry-sum adders, the ftcb adders (4 to 64 qubits), modular adders, multiplier and reduction, `mod5-4` |
| `clifford/` | 22 | random Clifford circuits by qubit count and CX ratio (`-cx25/50/75`) or depth (`-d50/100`), `clifford-complete-4`, `random-500` |
| `ftcb/` | 47 | fault-tolerant compilation benchmarks from 4 to 200 qubits: Fermi-Hubbard, Heisenberg and Ising on chains, square and triangular lattices, QFT, QPE on Hubbard, HHL, QSVT banded circulant |
| `gf-mult/` | 15 | GF(2^N) Galois-field multipliers, `gf2eN-mult` for N from 4 to 256 and three small `gfmult` variants |
| `graphstate/` | 12 | graph states on 4, 8 and 16 qubits: chain, complete, dense, grid, linear, random, ring |
| `parallel-t/` | 8 | layers of parallel T gates (`par-t-N`) and small T-ordering fixtures (chain, CX commutation, CX dependency, die) |
| `qualtran/` | 23 | Qualtran bloqs with their call tree: adders, modular add, multiply and exponentiation, GF(2^N) multiply and invert, comparison (`lte`), multi-controlled AND, QROM, Hubbard SELECT |
| `synthlr/` | 5 | SynthesizeLR worst-case linear-reversible CNOT circuits on 4 to 12 qubits |
| `t-factory/` | 8 | magic-state injection fixtures: chain, layer, pair, sandwich, smoke, Clifford-3 |
| `toffoli/` | 8 | Toffoli `tof-N` and Barenco-Toffoli `barenco-tof-N` for N in 3, 4, 5, 10 |

miniflash compiles Clifford+T: h/cx/x/s/sdg/t/tdg directly,
z/y/cz/swap/id/rz(k*pi/2)/ccx by auto-expansion. Circuits with gates
outside that set (arbitrary-angle rz etc.) raise at parse.

## Hierarchical circuits

Each circuit in `qualtran/` comes twice: `<name>.qasm`, flat Clifford+T like every
other benchmark, and `<name>.json`, the same circuit as definitions that call one
another, one per bloq. Loading the JSON gives the pair DAG and, for every call, the
qubits it acts on and the pairs it spans:

    import miniflash as flash

    h = flash.Hierarchy.load("benchmarks/qualtran/modadd-4.json")
    h.problem                       # what the compiler takes
    h.calls("modadd")               # {'add': 1, 'addk': 1, 'addk_2': 1, 'lineardepthgreaterthan': 1}
    h.counts("add")                 # gates of one call, callees included
    h.defs["add"].preds()           # dependencies between the items of one body
    for node in h.tree.walk():      # one node per call
        node.name, node.qubits, node.pairs

Regenerate or add sizes with the system `python3`, which has qualtran:

    python3 scripts/qualtran_bloqs.py --list
    python3 scripts/qualtran_bloqs.py add-64 gf2mul-12

A measurement-based uncomputation (And†) keeps its conditional CZ as an ordinary
gate, so the circuits are a worst case and not a unitary; the JSON marks the
condition. Rotations that are not multiples of pi/4 are refused.
