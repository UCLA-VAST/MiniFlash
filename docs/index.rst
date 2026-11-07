MiniFlash
=========

**An ultra fast Clifford+T → lattice-surgery compiler.**

MiniFlash turns an OpenQASM 2.0 circuit into a **tile program**: which tiles hold
qubits, magic states and routes, and which tiles merge at each step. It minimizes
spacetime volume (height × width × steps, walks included), checks every program
with an independent verifier, and renders it as a glTF scene you can open in any
3-D viewer.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Install and compile your first circuit into a verified tile program
      and a glTF scene.

   .. grid-item-card:: How it works
      :link: pipeline
      :link-type: doc

      The ladder of passes, from the pair DAG through placement, routing
      and spacing to the step graphs.

   .. grid-item-card:: API reference
      :link: api/index
      :link-type: doc

      Every module of the package, with signatures and type hints.

.. toctree::
   :hidden:
   :caption: Getting started

   quickstart
   configuration

.. toctree::
   :hidden:
   :caption: Inside

   pipeline
   benchmarks

.. toctree::
   :hidden:
   :caption: Reference

   api/index

Citation
--------

If you use MiniFlash in your research, please cite
`arXiv:2610.30747 <https://arxiv.org/abs/2610.30747>`_:

.. code-block:: bibtex

   @article{wang2026miniflash,
     title   = {Demand-Driven Routing Space for Lattice-Surgery Compilation},
     author  = {Wang, Hanyu and Lin, Wan-Hsuan and Chen, Zihan and Zhang, Eddy Z. and Cong, Jason},
     journal = {arXiv preprint arXiv:2610.30747},
     year    = {2026},
     url     = {https://arxiv.org/abs/2610.30747}
   }
