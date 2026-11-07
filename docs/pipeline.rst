How it works
============

A circuit goes through four objects. :class:`~miniflash.circuit.Problem` is the
input, :class:`~miniflash.mapping.Mapping` and the routed
:data:`~miniflash.route.Schedule` are the compiler's intermediates, and
:class:`~miniflash.program.Program` is the output that the verifier checks and the
glTF backend draws.

The problem
-----------

A Clifford+T circuit becomes a DAG of ordered **pairs**. A pair is ``(z, x)``: the
party attached through its Z boundary at the top or bottom, and the party attached
through its X boundary at the left or right. A CX is ``(control, target)`` and a T
gate is ``(qubit, MAGIC)`` with the magic state as the X party. Single-qubit
Cliffords are free in the execution model, so the reader drops them but counts what
it dropped. Dependencies are program order per qubit; MAGIC never induces one,
because every magic state is fresh.

The ladder
----------

The compiler is one class configured by components added one at a time. Every
configuration runs the same three passes, *place*, then *schedule* or *route*, then
*space*, and the presets of :func:`~miniflash.compiler.make` name the steps:

.. list-table::
   :header-rows: 1
   :widths: 16 70

   * - preset
     - what it adds
   * - ``vanilla``
     - one pair per step, in program order, along the shortest legal route
   * - ``dependency``
     - ASAP scheduling: every ready pair whose shortest free route exists runs in the step
   * - ``routing``
     - the criticality-driven router with lane scoring on the fixed grid
   * - ``spacing``
     - on-demand spacing: the router closes partitions by budget, each is compacted, qubits walk between them
   * - ``placement``
     - a seed from the interaction graph annealed on the paired-lane array instead of index order
   * - ``layout``
     - the shape of the site array, predicted from the DAG or chosen among the smallest-area shapes

**Placement** (:mod:`miniflash.synthesis.placement`). Index order on the sparse
square is the reference point. Otherwise a seed from the interaction graph,
spectral by default or one of the placers in :mod:`miniflash.synthesis.initial`, is
annealed by swapping qubits between sites, empty ones included, to lower a routing
proxy summed over the ASAP layers of the DAG: the collisions a layer scan detects,
the rows and columns loaded beyond the site lines, and the tentative route tiles.
Only the layers touching the two swapped qubits are re-scored.

**Routing** (:mod:`miniflash.synthesis.routing`). Pairs are ordered by the longest
remaining dependency chain. A small frontier is selected by branch and bound over
the route candidates, maximising criticality placed minus the cost of newly loaded
lines; a larger one greedily by route length plus line cost. A partition
accumulates the rows and columns its routes load beyond the site lines and closes
when no ready pair fits within the budget. Routes are chains of straight segments
through free tiles (:mod:`miniflash.route`): a straight route, an L, or a detour of
three segments.

**Spacing** (:mod:`miniflash.synthesis.spacing`). A row is load-bearing when it
holds a qubit tile, a magic tile some route consumes, or a segment endpoint;
removing every other row and column only shortens the segments that cross it. Each
partition is compacted on its own, the chip is the largest partition height by the
largest width, and between two partitions each qubit walks one tile per axis per
step, which never collides because both mappings realise the same site order.

**Layout** (:mod:`miniflash.synthesis.layout`). The statistics policy prices each
site-row count by full grid area times the layer capacity and compiles only its
preferred shape; ``layout=N`` compiles the N smallest-area shapes plus the squarest
one and keeps the cheapest program.

Magic supply
------------

Without a factory, magic states are unlimited. A :class:`~miniflash.factory.Factory`
alone puts one factory behind every magic tile, so a tile yields a state every
``period`` steps; ``factories=F`` shares F of them over the whole chip, F states per
period, delivered through the rings of :mod:`miniflash.supply`. The router and the
verifier enforce the same limits, and :func:`~miniflash.factory.charge` prices a
finished program for the factory volume behind it.

Hierarchies
-----------

A :class:`~miniflash.hierarchy.Hierarchy` is a circuit together with the call tree
it was built from. :meth:`~miniflash.compiler.Compiler.solve_hierarchy` compiles it
bottom-up on one mapping (:mod:`miniflash.synthesis.hierarchical`): each distinct
leaf is routed once, each parent is composed from its cached children, and only the
root enforces the magic supply and runs the spacing pass.

The program and the verifier
----------------------------

A :class:`~miniflash.program.Program` is one sparse step graph per time step: the
non-free tiles as vertices, each a qubit, a magic state or a route tile with its
pair id, and the merges between them as edges. :func:`~miniflash.verify.verify`
reads only those graphs: every qubit sits on exactly one tile, edges join
4-neighbours, every component realises exactly one pair through the right sides,
each pair runs once after its predecessors, and a qubit moves at most one tile per
axis per step. :func:`~miniflash.gltf.write_gltf` lowers the same graphs to pipes
on a cube lattice.

Resynthesis
-----------

:mod:`miniflash.resynthesis` cuts closed windows out of a finished program and asks
lassynth for a shallower cell that realises the same merges. It needs the SAT
toolchain and is driven by ``RENDER_WINDOWS`` in the constants.
