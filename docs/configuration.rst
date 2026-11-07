Configuration
=============

Every knob of the compiler lives in ``miniflash/constant.py``. The driver reads
nothing else, and the library's defaults read from there, so a call to
:class:`~miniflash.compiler.Compiler` or :func:`~miniflash.compiler.make` may still
override any of them for one run. The groups are:

* **Magic-state factory**: one factory is ``FACTORY_HEIGHT`` x ``FACTORY_WIDTH``
  tiles and yields one state every ``FACTORY_PERIOD`` steps. ``LIMITED_SUPPLY``
  turns the supply model on; with it, ``SHARED_FACTORIES`` factories around the
  chip feed it through delivery rings, or 0 puts one factory behind every magic
  tile. Off, magic states are unlimited.
* **Solver**: the ladder step the driver compiles with, the layout policy and how
  a hierarchy is placed.
* **Placement**: the initial placement and its seed, the anneal's proposals per
  qubit, temperatures and proxy weights, and the parameters of the GORDIAN,
  min-cut, APlace and mPL seeds.
* **Routing, spacing and layout**: the lane penalty, the frontier size the exact
  selection handles, the magic tiles tried per T pair, and the lanes a partition
  may load beyond its site lines.
* **Resynthesis**: the lassynth window budget, the slab cost weights and the
  kissat timeouts and cache.
* **Output**: the output directory, what the driver writes, and the glTF geometry.

.. literalinclude:: ../miniflash/constant.py
   :language: python
