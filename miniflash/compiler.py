# The compiler: one class configured by a ladder of components added one at a time, vanilla,
# dependency, routing, spacing, placement and layout. Every configuration runs the same three
# passes from miniflash.synthesis: place, then schedule or route, then space. Layout selection
# repeats them per candidate shape and keeps the lowest final volume. The passes are timed into
# `profile`, one entry per tried shape. Two controls fall outside the ladder: spacing="whole"
# compacts the whole schedule once, and spacing=True without routing cuts a finished schedule by budget.
from __future__ import annotations

import functools
import time
from typing import Optional, Union

from . import constant
from .circuit import Problem
from .factory import Factory, supply_limits
from .hierarchy import Hierarchy
from .mapping import Layout, Mapping
from .program import Program
from .route import Schedule
from .synthesis import hierarchical, routing
from .synthesis.layout import Shape, candidate_shapes, stats_shape
from .synthesis.placement import Placement, Workload, place
from .synthesis.schedule import schedule
from .synthesis.spacing import Partition, budgets, cut, space

LADDER = ("vanilla", "dependency", "routing", "spacing", "placement", "layout")
WEIGHTS = (constant.ANNEAL_COLLISION_WEIGHT, constant.ANNEAL_LANE_WEIGHT, constant.ANNEAL_LENGTH_WEIGHT)


class Compiler:
    def __init__(self, dependency: bool = True, routing: bool = False, spacing: Union[bool, str] = False,
                 placement: Union[bool, Placement] = False, layout: Union[None, str, int] = None,
                 shape: Optional[Shape] = None, extra: int = constant.SPACING_EXTRA_LANES,
                 share: float = constant.ROUTING_SHARE_PENALTY, row_budget: Optional[int] = None,
                 col_budget: Optional[int] = None, exact: int = constant.ROUTING_EXACT_FRONTIER,
                 factory: Optional[Factory] = None, factories: int = 0, seed: int = constant.PLACEMENT_SEED,
                 iters: Optional[int] = None, anneal: bool = True, weights: tuple[float, float, float] = WEIGHTS,
                 init: str = constant.INITIAL_PLACEMENT):
        # dependency False schedules one pair per step, True ASAP; routing replaces both with the
        # criticality-driven router. spacing True partitions by budget and walks between partitions,
        # "whole" compacts once. placement True seeds and anneals on the paired-lane array, a Placement
        # fixes the mapping, False keeps index order on the sparse square. layout None uses `shape`,
        # "stats" the shape predicted from the DAG, an int compares that many smallest-area shapes.
        # factory alone puts one factory behind every magic tile; factories = F shares F of them.
        self.dependency = dependency
        self.routing = routing
        self.spacing = spacing
        self.placement = placement
        self.layout_mode = layout
        self.shape_option = shape
        self.extra = extra
        self.share = share
        self.row_budget = row_budget
        self.col_budget = col_budget
        self.exact = exact
        self.factory = factory
        self.factories = factories
        self.seed = seed
        self.iters = iters
        self.anneal = anneal
        self.weights = weights
        self.init = init

        # results of the last solve, of the selected shape when several were tried
        self.shape: Optional[Shape] = None
        self.layout: Optional[Layout] = None
        self.mapping: Optional[Mapping] = None
        self.partitions: list[Partition] = []
        self.windows: list[tuple[int, int]] = []
        self.keep_rows: list[int] = []
        self.keep_cols: list[int] = []
        self.initial_cost = 0.0
        self.final_cost = 0.0
        self.tried: list[dict] = []
        self.profile: list[dict] = []
        self.hierarchy_stats: dict = {}
        self.hierarchy_plan: Optional[hierarchical.Plan] = None
        self.hierarchy_templates: list[Schedule] = []

    @property
    def name(self) -> str:
        # The ladder label of the last component, with the non-default options as suffixes.
        if self.layout_mode is not None:
            label = "layout"
        elif self.placement:
            label = "placement" if self.spacing is True and self.routing else "placed"
        elif self.spacing == "whole":
            label = "compact"
        elif self.spacing:
            label = "spacing" if self.routing else "staged"
        elif self.routing:
            label = "routing"
        else:
            label = "dependency" if self.dependency else "vanilla"

        suffix = ""
        if isinstance(self.layout_mode, int):
            suffix += f"-shapes{self.layout_mode}"
        if self.init != constant.INITIAL_PLACEMENT and self.placement is True:
            suffix += f"-{self.init}"
        if self.seed != constant.PLACEMENT_SEED and self.placement:
            suffix += f"-s{self.seed}"
        if self.factory is not None and self.routing:
            suffix += f"-{self.factory.height}x{self.factory.width}x{self.factory.period}"
        if self.factories and self.routing:
            suffix += f"-f{self.factories}"
        if self.extra != constant.SPACING_EXTRA_LANES and self.spacing is True:
            suffix += f"-extra{self.extra}"
        if self.share != constant.ROUTING_SHARE_PENALTY and self.routing:
            suffix += f"-share{self.share:g}"
        if self.hierarchy_stats:
            suffix += "-hierarchical"
        return label + suffix

    def map(self, problem: Problem, shape: Optional[Shape] = None, workload: Optional[Workload] = None) -> Mapping:
        if isinstance(self.placement, Placement):
            placed = self.placement
        else:
            placed = place(problem, shape, bool(self.placement), self.seed, self.iters, self.anneal, self.weights,
                           workload, self.init)
        self.mapping = placed.mapping
        self.layout = placed.layout
        self.initial_cost = placed.initial_cost
        self.final_cost = placed.final_cost
        return placed.mapping

    def budgets(self, mapping: Mapping) -> tuple[int, int]:
        # Rows and columns a partition may retain, the whole grid when nothing is partitioned.
        if self.spacing is True:
            return budgets(self.layout, self.extra, self.row_budget, self.col_budget)
        return (self.row_budget if self.row_budget is not None else mapping.height,
                self.col_budget if self.col_budget is not None else mapping.width)

    def route(self, problem: Problem, mapping: Mapping) -> Schedule:
        # The router also records the partition windows it closed.
        self.windows = []
        if self.routing:
            steps, self.windows = routing.route(problem, mapping, *self.budgets(mapping), self.share, self.exact,
                                                *supply_limits(self.factory, self.factories))
            return steps
        return schedule(problem, mapping, self.dependency)

    def build(self, problem: Problem, mapping: Mapping, steps: Schedule) -> Program:
        self.partitions = []
        self.keep_rows = []
        self.keep_cols = []
        if self.spacing is True:
            program, self.partitions = space(problem, mapping, steps, "partition",
                                             windows=self.windows if self.routing else None, budgets=self.budgets(mapping))
        elif self.spacing == "whole":
            program, (self.keep_rows, self.keep_cols) = space(problem, mapping, steps, "whole")
        else:
            program, _ = space(problem, mapping, steps)
        return program

    def solve_shape(self, problem: Problem, shape: Optional[Shape] = None,
                    cells: Optional[hierarchical.Plan] = None, workload: Optional[Workload] = None) -> Program:
        started = time.perf_counter()
        mapping = self.map(problem, shape, workload)
        mapped = time.perf_counter()

        self.hierarchy_plan = cells
        self.hierarchy_templates = []
        if cells is None:
            steps = self.route(problem, mapping)
            compose_s = 0.0
            windows_s = 0.0
            self.hierarchy_stats = {}
        else:
            # Descendants are reusable templates; only the root consumes the chip's supply, so no
            # child starts a fresh factory clock.
            compose_s = 0.0
            templates = self.hierarchy_templates
            for index, cell in enumerate(cells.cells):
                supply = supply_limits(self.factory, self.factories) if index == cells.root else (1, 0, constant.FACTORY_PERIOD)
                if cell.calls:
                    started = time.perf_counter()
                    templates.append(hierarchical.compose(cell.problem, mapping, cell.calls, templates, *supply))
                    compose_s += time.perf_counter() - started
                else:
                    templates.append(routing.route(cell.problem, mapping, mapping.height, mapping.width,
                                                   self.share, self.exact, *supply)[0])
            steps = templates[cells.root] if cells.root is not None else []

            # Cell windows cannot be reused, since overlap and rerouting change the loaded lines.
            composed = time.perf_counter()
            self.windows = (cut(mapping, steps, *self.budgets(mapping)) or [(0, 0)]) if self.spacing is True else []
            windows_s = time.perf_counter() - composed
            self.hierarchy_stats = cells.stats()
        routed = time.perf_counter()

        program = self.build(problem, mapping, steps)
        built = time.perf_counter()
        self.shape = shape if shape is not None else (self.layout.site_rows, self.layout.site_cols)
        self.profile.append({"map_s": mapped - started, "route_s": routed - mapped - compose_s - windows_s,
                             "compose_s": compose_s, "windows_s": windows_s, "compact_s": built - routed,
                             "total_s": built - started,
                             "shape": list(self.shape), "grid": [program.height, program.width],
                             "steps": len(program.steps), "volume": program.volume, **self.hierarchy_stats})
        return program

    def solve(self, problem: Problem) -> Program:
        return self._solve(problem)

    def solve_hierarchy(self, hierarchy: Hierarchy, placement_mode: str = constant.HIERARCHY_PLACEMENT) -> Program:
        # Route the leaves, compose and cache each parent, then space at the root. Needs routing.
        # "weighted" places from leaf layers weighted by expanded call counts, "flat" from the full DAG.
        started = time.perf_counter()
        cells = hierarchical.extract(hierarchy)
        workload = cells.workload(hierarchy.n_qubits) if placement_mode == "weighted" else None
        extract_s = time.perf_counter() - started

        program = self._solve(hierarchy.problem, cells, workload)
        self.hierarchy_stats.update(placement_mode=placement_mode, extract_s=extract_s)
        return program

    def _solve(self, problem: Problem, cells: Optional[hierarchical.Plan] = None,
               workload: Optional[Workload] = None) -> Program:
        self.tried = []
        self.profile = []
        if self.layout_mode is None:
            return self.solve_shape(problem, self.shape_option, cells, workload)

        if self.layout_mode == "stats":
            shapes = [stats_shape(problem)]
        else:
            shapes = candidate_shapes(problem.n_qubits, self.layout_mode, squarest=True)
        best = None
        for shape in shapes:
            program = self.solve_shape(problem, shape, cells, workload)
            self.tried.append({"shape": shape, "grid": (program.height, program.width), "steps": len(program), "volume": program.volume})
            if best is None or program.volume < best[0].volume:
                best = (program, shape, self.layout, self.mapping, self.partitions, self.windows,
                        self.keep_rows, self.keep_cols, self.initial_cost, self.final_cost, self.hierarchy_stats,
                        self.hierarchy_templates)
        (program, self.shape, self.layout, self.mapping, self.partitions, self.windows,
         self.keep_rows, self.keep_cols, self.initial_cost, self.final_cost, self.hierarchy_stats,
         self.hierarchy_templates) = best
        return program


# Compiler keyword arguments by preset name: the ladder, then the development controls.
PRESETS = {
    "vanilla": dict(dependency=False),
    "dependency": dict(),
    "routing": dict(routing=True),
    "spacing": dict(routing=True, spacing=True),
    "placement": dict(routing=True, spacing=True, placement=True),
    "layout": dict(routing=True, spacing=True, placement=True, layout="stats"),
    "compact": dict(spacing="whole"),
    "staged": dict(spacing=True),
    "placed": dict(placement=True),
}


def make(name: str, shapes: Union[None, str, int] = None, **options) -> Compiler:
    # `shapes` replaces the preset's layout mode, "stats" or a count; None and "default" keep it.
    arguments = dict(PRESETS[name])
    if shapes not in (None, "default") and "layout" in arguments:
        arguments["layout"] = shapes
    arguments.update(options)
    return Compiler(**arguments)


SOLVERS = {name: functools.partial(make, name) for name in PRESETS}
