Quickstart
==========

Install
-------

.. code-block:: bash

   git clone https://github.com/UCLA-VAST/MiniFlash.git
   cd MiniFlash
   pip install -e '.[test]'
   python -m pytest -q tests

MiniFlash is pure Python and needs only qiskit, for reading OpenQASM, and numpy.
The optional resynthesis of windows with a SAT solver additionally needs
``lassynth``, ``stim``, ``z3`` and a ``kissat`` binary; the tests that use them
skip when they are missing.

First compile
-------------

.. code-block:: bash

   python main.py benchmarks/algorithms/ghz8.qasm -o ghz8.gltf

The driver prints one CSV row per circuit, every row checked by the independent
verifier, and writes the glTF scene. Open it in any viewer, such as the
`three.js viewer <https://gltf-viewer.donmccurdy.com/>`_. Time runs upward, two
layers per step:

* qubits are red pillars;
* a CNOT splits at its first turn, where an ancilla climbs: the half from the
  control's Z side sits on the lower layer in blue, the half into the target's
  X side on the upper layer in red;
* T merges end at green magic states;
* walks between partitions are orange.

Several circuits at once print a table; the scenes land in the output directory,
``out/`` by default, one per circuit:

.. code-block:: bash

   python main.py benchmarks/toffoli/*.qasm > toffoli.csv
   python main.py benchmarks/qualtran/add-4.json

A hierarchy ``.json`` from ``benchmarks/qualtran/`` compiles bottom-up through its
call tree, each distinct subroutine routed once and composed into its callers; the
same circuit as a flat ``.qasm`` compiles as one DAG. The driver picks the mode
from the suffix.

The CSV columns are ``circuit``, ``qubits``, ``pairs``, ``depth`` (the DAG's
layers), ``grid`` (height x width), ``steps``, ``walk_steps``, ``volume`` (steps
x height x width) and the ``compile_s`` and ``verify_s`` timings.

Command line
------------

``main.py`` takes circuit paths and one option:

.. list-table::
   :header-rows: 1
   :widths: 24 60

   * - argument
     - meaning
   * - ``circuits``
     - ``.qasm`` circuits or hierarchy ``.json`` files
   * - ``-o / --output``
     - the output directory, or a ``.gltf`` path when there is one circuit

Everything else, the solver step, the magic-state factory, the placement seed and
the lassynth budgets, is a constant in ``miniflash/constant.py``; see
:doc:`configuration`.

As a library
------------

.. code-block:: python

   import miniflash as flash

   problem = flash.read_qasm("benchmarks/toffoli/tof-3.qasm")
   program = flash.make("layout").solve(problem)
   report = flash.verify(problem, program)
   flash.write_gltf(program, "tof3.gltf")

``flash.make(name, **options)`` builds a :class:`~miniflash.compiler.Compiler`
from a preset, one step of the ladder (``vanilla``, ``dependency``, ``routing``,
``spacing``, ``placement``, ``layout``), and any keyword of ``Compiler`` overrides
the constant for that one call: ``flash.make("layout", seed=3, init="gordian")``.
:func:`~miniflash.verify.verify` raises :class:`~miniflash.verify.VerifyError`
when a program does not execute its problem and otherwise returns a
:class:`~miniflash.verify.Report` with the grid, the steps and the step each pair
ran at.
