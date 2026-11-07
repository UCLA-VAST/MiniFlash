# miniflash compiles Clifford+T OpenQASM 2.0 into a lattice-surgery tile program:
# parse to a Problem (the pair DAG), Compiler (place, route, space) to a Program (one step graph
# per time step), verify, write_gltf.
from .circuit import MAGIC, Builder, Pair, Problem, from_circuit, read_qasm, to_dot
from .parse import parse
from .mapping import Layout, Mapping
from .route import Route, Segment, check_route
from .program import Program, Step, Vertex
from .synthesis import Placement, candidate_shapes, compact, place, schedule, space, stats_shape
from .compiler import LADDER, PRESETS, SOLVERS, Compiler, make
from .synthesis.placement import INITS
from .factory import Bill, Factory, charge
from .supply import Supply, plan
from .verify import Report, VerifyError, verify
from .gltf import write_gltf
from .hierarchy import Def, Hierarchy, Node

__version__ = "1.0.0a0"

__all__ = ["MAGIC", "Builder", "Pair", "Problem", "from_circuit", "read_qasm", "to_dot", "parse", "Layout", "Mapping",
           "Route", "Segment", "check_route", "Program", "Step", "Vertex", "Placement", "place", "schedule", "compact",
           "space", "candidate_shapes", "stats_shape", "LADDER", "INITS", "PRESETS", "SOLVERS", "Compiler", "make", "Bill",
           "Factory", "charge", "Supply", "plan", "Report", "VerifyError", "verify", "write_gltf", "Def", "Hierarchy", "Node"]
