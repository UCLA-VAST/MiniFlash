Benchmarks
==========

Compiled **spacetime volume** on 44 circuits, against **TopoLS** (Zhou
et al., `arXiv:2601.23109 <https://arxiv.org/abs/2601.23109>`_) and
**DasCot** (Molavi et al., `arXiv:2311.18042
<https://arxiv.org/abs/2311.18042>`_). Below 1× = MiniFlash smaller.

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 16

   * - family
     - circuits
     - geomean vs TopoLS
     - geomean vs DasCot
   * - GHZ state preparation
     - 7
     - 0.61×
     - 0.20×
   * - Bernstein–Vazirani
     - 8
     - 0.70×
     - 0.27×
   * - Deutsch–Jozsa
     - 8
     - 1.41×
     - 0.92×
   * - random Clifford
     - 9
     - 0.73×
     - 0.81×
   * - graph states
     - 12
     - 0.63×
     - 0.75×
   * - **all**
     - **44**
     - **0.77×**
     - **0.53×**

Four of the five families favor MiniFlash; the driver on both sides is
**cell granularity**. Wins come from whole-circuit cells that amortize
all fixed overhead; losses appear exactly where the SAT instance
exceeds its budget and the circuit splits into stitched fragments.

.. note::

   Volume = tile bounding box, the same unit in all three tools.
   Identical QASM inputs; TopoLS under a 180 s wall (misses "—",
   bv-100 from its fast profile). MiniFlash runs use ``--engine sat``
   with a warm cell cache and the default 600 s per-region budget.
   Measured 2026-08-25.

GHZ State Preparation
---------------------

The advantage widens with size and then stabilizes — one whole-circuit
cell, constant overhead amortized.

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 16 16 16

   * - circuit
     - MiniFlash
     - TopoLS
     - DasCot
     - vs TopoLS
     - vs DasCot
   * - ``ghz3``
     - 45
     - 42
     - 98
     - 1.07×
     - 0.46×
   * - ``ghz4``
     - 54
     - 81
     - 147
     - 0.67×
     - 0.37×
   * - ``ghz6``
     - 72
     - 135
     - 405
     - 0.53×
     - 0.18×
   * - ``ghz8``
     - 120
     - 135
     - 567
     - 0.89×
     - 0.21×
   * - ``ghz10``
     - 144
     - 315
     - 1,089
     - 0.46×
     - 0.13×
   * - ``ghz12``
     - 168
     - 378
     - 1,331
     - 0.44×
     - 0.13×
   * - ``ghz16``
     - 216
     - 486
     - 1,815
     - 0.44×
     - 0.12×

Bernstein–Vazirani
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 16 16 16

   * - circuit
     - MiniFlash
     - TopoLS
     - DasCot
     - vs TopoLS
     - vs DasCot
   * - ``bv-16``
     - 216
     - 648
     - 1,815
     - 0.33×
     - 0.12×
   * - ``bv-20``
     - 552
     - 1,584
     - 3,211
     - 0.35×
     - 0.17×
   * - ``bv-30``
     - 891
     - 2,295
     - 6,525
     - 0.39×
     - 0.14×
   * - ``bv-40``
     - 3,672
     - 3,402
     - 11,271
     - 1.08×
     - 0.33×
   * - ``bv-50``
     - 6,240
     - 5,103
     - 17,689
     - 1.22×
     - 0.35×
   * - ``bv-60``
     - 7,425
     - 7,533
     - 21,299
     - 0.99×
     - 0.35×
   * - ``bv-80``
     - 17,976
     - —
     - 34,839
     - —
     - 0.52×
   * - ``bv-100``
     - 29,160
     - 20,196
     - 52,371
     - 1.44×
     - 0.56×

Deutsch–Jozsa
-------------

The one family that loses to TopoLS (1.41×): star connectivity
fragments into a serial chain of cells and the stitching depth adds up.
Against DasCot it still wins narrowly (0.92×).

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 16 16 16

   * - circuit
     - MiniFlash
     - TopoLS
     - DasCot
     - vs TopoLS
     - vs DasCot
   * - ``dj-16``
     - 612
     - 486
     - 567
     - 1.26×
     - 1.08×
   * - ``dj-20``
     - 966
     - 594
     - 1,089
     - 1.63×
     - 0.89×
   * - ``dj-30``
     - 2,244
     - 1,071
     - 1,694
     - 2.10×
     - 1.32×
   * - ``dj-40``
     - 2,064
     - 1,890
     - 3,211
     - 1.09×
     - 0.64×
   * - ``dj-50``
     - 3,888
     - 2,916
     - 4,056
     - 1.33×
     - 0.96×
   * - ``dj-60``
     - 3,402
     - 4,464
     - 6,525
     - 0.76×
     - 0.52×
   * - ``dj-80``
     - 12,528
     - 8,118
     - 11,271
     - 1.54×
     - 1.11×
   * - ``dj-100``
     - 20,928
     - 10,098
     - 17,689
     - 2.07×
     - 1.18×

Random Clifford
---------------


.. list-table::
   :header-rows: 1
   :widths: 22 16 16 16 16 16

   * - circuit
     - MiniFlash
     - TopoLS
     - DasCot
     - vs TopoLS
     - vs DasCot
   * - ``4q-cx25``
     - 54
     - 243
     - 245
     - 0.22×
     - 0.22×
   * - ``4q-cx50``
     - 54
     - 243
     - 392
     - 0.22×
     - 0.14×
   * - ``4q-cx75``
     - 54
     - 351
     - 441
     - 0.15×
     - 0.12×
   * - ``8q-cx25``
     - 144
     - 540
     - 243
     - 0.27×
     - 0.59×
   * - ``8q-cx50``
     - 855
     - 630
     - 567
     - 1.36×
     - 1.51×
   * - ``8q-cx75``
     - 918
     - 810
     - 891
     - 1.13×
     - 1.03×
   * - ``16q-cx25``
     - 1,248
     - 891
     - 605
     - 1.40×
     - 2.06×
   * - ``16q-cx50``
     - 5,940
     - 1,296
     - 968
     - 4.58×
     - 6.14×
   * - ``16q-cx75``
     - 5,940
     - 2,025
     - 1,694
     - 2.93×
     - 3.51×

Graph States
------------

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 16 16 16

   * - circuit
     - MiniFlash
     - TopoLS
     - DasCot
     - vs TopoLS
     - vs DasCot
   * - ``gs4-chain``
     - 54
     - 162
     - 147
     - 0.33×
     - 0.37×
   * - ``gs4-ring``
     - 54
     - 189
     - 196
     - 0.29×
     - 0.28×
   * - ``gs4-dense``
     - 360
     - 243
     - 196
     - 1.48×
     - 1.84×
   * - ``gs4-complete``
     - 504
     - 297
     - 245
     - 1.70×
     - 2.06×
   * - ``gs8-ring``
     - 120
     - 360
     - 648
     - 0.33×
     - 0.19×
   * - ``gs8-linear``
     - 396
     - 360
     - 567
     - 1.10×
     - 0.70×
   * - ``gs8-grid2x4``
     - 546
     - 405
     - 324
     - 1.35×
     - 1.69×
   * - ``gs8-rand50``
     - 363
     - 1,215
     - 729
     - 0.30×
     - 0.50×
   * - ``gs8-rand75``
     - 858
     - 1,935
     - 891
     - 0.44×
     - 0.96×
   * - ``gs8-complete``
     - 594
     - 2,655
     - 1,053
     - 0.22×
     - 0.56×
   * - ``gs16-linear``
     - 1,254
     - —
     - 1,815
     - —
     - 0.69×
   * - ``gs16-grid``
     - 3,078
     - 1,620
     - 1,452
     - 1.90×
     - 2.12×

Sparse Long-Range Circuits (SynthLR)
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 16 16 16 16

   * - circuit
     - MiniFlash
     - manual
     - TopoLS
     - DasCot
     - vs manual
     - vs DasCot
   * - ``synthlr-4``
     - 54
     - 42
     - 351
     - 539
     - 1.29×
     - 0.10×
   * - ``synthlr-6``
     - 96
     - 132
     - 1,125
     - 1,944
     - 0.73×
     - 0.05×
   * - ``synthlr-8``
     - 120
     - 270
     - 1,980
     - 3,321
     - 0.44×
     - 0.04×
   * - ``synthlr-10``
     - 144
     - 456
     - —
     - 7,502
     - 0.32×
     - 0.02×
   * - ``synthlr-12``
     - 168
     - 690
     - —
     - 10,648
     - 0.24×
     - 0.02×

T-Dense Circuits (Galois-Field Multiplication)
----------------------------------------------

Injection-dominated circuits (112–448 T gates): the fixed-column mode
(``--fixed-columns --engine sat --factory external --die W --factories 2``,
per-size widths 24/32/40/48/56 — interleave-merged joint cells, packed
injections, distributed magic entry) compares to DasCot two ways.
**As reported**, DasCot counts every timestep as one unit of time, a
CNOT included — geomean **1.06×** against that accounting, with
``gf2e4-mult`` and ``gf2e6-mult`` below parity. But a CNOT cannot
complete in one merge round: its :math:`M_{ZZ}` and :math:`M_{XX}` are
anticommuting and must be sequential, so a timestep that routes a CNOT
physically lasts two rounds while a step holding only T injections (a
single :math:`M_{ZZ}`) lasts one. **Converted per step** from DasCot's
own schedules (74–207 of its steps per circuit route a CNOT), the
geomean is **0.60×** — every size below parity. The solver-free packer
(``--engine packing --t-layers``, seconds of compile time) matches
both (1.05× reported, 0.59× converted). TopoLS times out on all of
them.

.. list-table::
   :header-rows: 1
   :widths: 22 10 14 14 12 14 12

   * - circuit
     - T count
     - MiniFlash
     - DasCot (reported)
     - vs reported
     - DasCot (converted)
     - vs converted
   * - ``gf2e4-mult``
     - 112
     - 11,634
     - 12,463
     - **0.93×**
     - 21,417
     - **0.54×**
   * - ``gf2e5-mult``
     - 175
     - 18,564
     - 16,456
     - 1.13×
     - 28,677
     - **0.65×**
   * - ``gf2e6-mult``
     - 252
     - 27,780
     - 28,899
     - **0.96×**
     - 50,869
     - **0.55×**
   * - ``gf2e7-mult``
     - 343
     - 37,260
     - 34,645
     - 1.08×
     - 62,699
     - **0.59×**
   * - ``gf2e8-mult``
     - 448
     - 51,090
     - 42,081
     - 1.21×
     - 77,064
     - **0.66×**
