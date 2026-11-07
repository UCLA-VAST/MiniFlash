API reference
=============

``import miniflash as flash`` exposes the public names at the package top level;
the modules below hold the implementations. There are no docstrings: each page
lists the signatures with their type hints, and the module header of the source,
reachable through the ``[source]`` links, states what the module does.

**Front end**, circuit to problem:

.. toctree::
   :maxdepth: 1

   circuit
   parse
   hierarchy

**Compiler**, problem to program:

.. toctree::
   :maxdepth: 1

   compiler
   mapping
   route
   placement
   initial
   schedule
   routing
   spacing
   layout
   hierarchical

**Output**, the program and what reads it:

.. toctree::
   :maxdepth: 1

   program
   verify
   factory
   supply
   gltf
   resynthesis
   constant
