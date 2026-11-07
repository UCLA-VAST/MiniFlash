Benchmarks
==========

``benchmarks/`` ships 206 OpenQASM circuits in eleven families. MiniFlash compiles
h, cx, x, s, sdg, t and tdg directly and expands z, y, cz, swap, id, rz at quarter
turns and ccx; a gate outside that set is refused at parse.

.. list-table::
   :header-rows: 1
   :widths: 16 8 60

   * - folder
     - count
     - circuits
   * - ``algorithms/``
     - 41
     - GHZ, BV, DJ, QFT, QPE, QAOA, Grover, Shor, HWB, Hamming, VQE, W state
   * - ``arithmetic/``
     - 17
     - adders (rc, csla, csum, qcla, vbe, mod, ftcb), ``qcla-com``, ``qcla-mod``, ``mod5-4``
   * - ``clifford/``
     - 22
     - random Clifford circuits by CX ratio and by depth, ``clifford-complete-4``, ``random-500``
   * - ``ftcb/``
     - 47
     - fault-tolerant compilation benchmarks: Fermi-Hubbard, Heisenberg, Ising, QFT, QPE, HHL, QSVT
   * - ``gf-mult/``
     - 15
     - GF(2^N) Galois-field multipliers
   * - ``graphstate/``
     - 12
     - graph states on 4, 8 and 16 qubits: chain, complete, dense, grid, linear, random, ring
   * - ``parallel-t/``
     - 8
     - layers of parallel T gates, ``par-t-N``
   * - ``synthlr/``
     - 5
     - SynthesizeLR worst-case linear-reversible CNOT circuits
   * - ``t-factory/``
     - 8
     - magic-state injection fixtures
   * - ``toffoli/``
     - 8
     - Toffoli ``tof-N`` and Barenco-Toffoli
   * - ``qualtran/``
     - 23
     - Qualtran bloqs with their call tree: adders, modular arithmetic up to exponentiation, GF(2^N) multiply and invert, comparison, QROM, Hubbard SELECT

Hierarchical circuits
---------------------

Each circuit in ``qualtran/`` comes twice: ``<name>.qasm``, flat Clifford+T like
every other benchmark, and ``<name>.json``, the same circuit as definitions that
call one another, one per bloq. Loading the JSON gives the pair DAG and, for every
call, the qubits it acts on and the pairs it spans:

.. code-block:: python

   import miniflash as flash

   h = flash.Hierarchy.load("benchmarks/qualtran/modadd-4.json")
   h.problem                       # what the compiler takes
   h.calls("modadd")               # callees of one definition and how often
   h.counts("add")                 # gates of one call, callees included
   for node in h.tree.walk():      # one node per call
       node.name, node.qubits, node.pairs

``scripts/qualtran_bloqs.py`` regenerates them or adds sizes; it needs ``qualtran``
and ``cirq``. A measurement-based uncomputation keeps its conditional CZ as an
ordinary gate, so the circuits are a worst case and not a unitary.
