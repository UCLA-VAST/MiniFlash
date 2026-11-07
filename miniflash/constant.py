# Every knob of the compiler in one place. main.py takes circuit paths and nothing else, and the
# library reads its defaults from here, so a call may still override any of them.
import os
from pathlib import Path

# ---- magic-state factory
# One factory is FACTORY_HEIGHT x FACTORY_WIDTH tiles and yields one state every FACTORY_PERIOD steps.
FACTORY_HEIGHT = 3
FACTORY_WIDTH = 3
FACTORY_PERIOD = 11
FACTORY_SIZING = "peak"   # how a program is charged for factories: peak, rate or tiles

# Whether the compiler and the verifier enforce that supply at all. Off, magic states are unlimited.
LIMITED_SUPPLY = False

# With a limited supply, this many factories around the chip feed it through delivery rings;
# 0 puts one factory behind every magic tile instead.
SHARED_FACTORIES = 0

# ---- the solver main compiles with
SOLVER = "layout"                  # a step of the ladder: vanilla, dependency, routing, spacing, placement, layout
LAYOUT_SHAPES = "stats"            # the shape predicted from the DAG, or an int comparing that many smallest-area shapes
HIERARCHY_PLACEMENT = "weighted"   # place a hierarchy from call-count-weighted cells, or "flat" from the full DAG

# ---- placement
INITIAL_PLACEMENT = "spectral"   # spectral, random, gordian, gordian-l, mincut, aplace or mpl
PLACEMENT_SEED = 0
INTERACTION_FLOOR = 1e-3         # weight between every qubit pair, so the interaction graph is connected

# The anneal swaps qubits between sites to lower a routing proxy summed over the ASAP layers.
ANNEAL_PROPOSALS_PER_QUBIT = 100
ANNEAL_START_TEMPERATURE = 0.05   # fraction of the initial proxy cost per layer, at least 1
ANNEAL_END_TEMPERATURE = 0.05
ANNEAL_COLLISION_WEIGHT = 6.0     # overlaps the layer scan detects
ANNEAL_LANE_WEIGHT = 6.0          # rows and columns loaded beyond the site lines
ANNEAL_LENGTH_WEIGHT = 0.25       # tentative route tiles
ROUTE_CACHE_PER_PAIR = 4          # recent tentative routes kept per pair, enough to restore a rejected proposal

# The initial placers besides spectral, in site units.
GORDIAN_L_MIN_DISTANCE = 0.25   # floor of the 1/|d| edge reweighting
GORDIAN_L_ROUNDS = 3            # reweighted solves per level
MINCUT_STARTS = 4               # random partitions refined by Kernighan-Lin per cut
APLACE_SMOOTHING = 0.5          # the log-sum-exp gamma
DENSITY_BIN = 0.5               # bins per site pitch, so 2 x 2 bins per site
DENSITY_START = 0.1             # the density weight starts at this fraction of the wirelength gradient
DENSITY_ROUNDS = 16             # doublings of the density weight
DENSITY_ITERS = 30              # conjugate gradient steps per round
DENSITY_OVERFLOW = 0.1          # done once the overflow is below this fraction of the cell area
CLOUD_SPREAD = 0.25             # the initial cloud spans this fraction of the array around its centre
MPL_COARSEST = 4                # qubits in the coarsest graph
MPL_CLUSTER_FRACTION = 0.25     # no cluster holds more than this fraction of the qubits
MPL_SHRINK = 0.9                # coarsening stops when a level keeps more than this fraction of the nodes
MPL_JITTER = 0.1                # noise added to a child's position when declustering

# ---- routing
ROUTING_SHARE_PENALTY = 4.0   # per row or column a route newly loads
ROUTING_EXACT_FRONTIER = 12   # frontiers up to this size are selected by branch and bound, larger ones greedily
ROUTING_NEAREST_MAGIC = 4     # magic tiles considered per T pair, by distance
ROUTE_CONNECTORS = 3          # connecting lines tried for three-segment routes

# ---- spacing and layout
SPACING_EXTRA_LANES = 6   # rows and columns a partition may load beyond its site lines
LAYOUT_CX_PER_ROW = 1.0   # CX pairs a site row carries per step in the shape estimate

# ---- resynthesis with lassynth
LAYERS_PER_STEP = 2
WINDOW_AREA = 16              # tiles one lassynth window may cover
WINDOW_STEPS = 4
WINDOW_PORTS = 12
WINDOW_CUT = "dp"             # cut the slabs by dynamic programming over a cost, or "greedy" by estimated gain
WINDOW_MIN_GAIN = 0           # the greedy cut drops slabs below this gain
SLAB_VOLUME_WEIGHT = 1.0      # per tile-step of the estimated slab
SLAB_HOLES_WEIGHT = 1.0       # per tile-step left empty above a window shallower than its slab
SLAB_CUTS_WEIGHT = 2.0        # per dependency edge a cut severs
SLAB_SKIP_WEIGHT = 4.0        # per step left outside every slab
LASSYNTH_TIMEOUT = 60         # seconds per kissat call
LASSYNTH_CHECK_TIMEOUT = 120  # seconds to check the tile program's own geometry
LASSYNTH_CACHE_DIR = Path(os.environ.get("MINIFLASH_CACHE", ".miniflash-cache")) / "resynthesis.v1"
KISSAT_BIN = os.environ.get("MINIFLASH_KISSAT_BIN")   # the kissat executable, found beside the interpreter or on the path when unset

# ---- output
OUTPUT_DIR = Path("out")   # main writes <circuit>.gltf and <circuit>.json here
WRITE_GLTF = True
WRITE_PROGRAM = False
RENDER_WINDOWS = None      # None, or draw the resynthesis windows: "overlay", "coarse", "cells" or "cells-inplace"
GLTF_STRIDE = 2            # cubes per tile along each axis
PIPE_AXIS_SCALE = 1.4
PIPE_CROSS_SCALE = 0.55
WINDOW_ALPHA = 0.35        # opacity of an overlaid window box
CELL_OUTLINE = 0.12        # outline width of a drawn cell
