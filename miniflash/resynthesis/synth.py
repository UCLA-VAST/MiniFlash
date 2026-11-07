# Runs lassynth on a window's spec and presses its depth. solve makes one SAT call through kissat under a wall-clock
# budget and keeps the answer on disk keyed by the spec, check plugs the tile program's own geometry into the spec so
# that sat means the spec and the tile model agree, and minimize climbs from one layer upwards until a depth is
# satisfiable, since unsatisfiable depths answer fast. cell returns the shallowest cell found for a window, or the tile
# program's own geometry as a cell when nothing shallower was found in time, and cells makes one per window of every
# slab. lassynth, z3, stim and kissat are needed only once a function here runs.
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .. import constant
from ..circuit import Problem
from ..program import Program
from .cell import Cell
from .spec import geometry, layers, spec
from .window import Slab, Window

SAT = "sat"
UNSAT = "unsat"
TIMEOUT = "timeout"


@dataclass(frozen=True)
class Result:
    status: str
    depth: int                     # layers asked for, the port layer not counted
    seconds: float
    lasre: Optional[dict] = None

    def __str__(self) -> str:
        return f"depth {self.depth}: {self.status} in {self.seconds:.2f}s"


def kissat_binary() -> str:
    # MINIFLASH_KISSAT_BIN when set, else the kissat beside the interpreter, else the first on the path.
    for candidate in (constant.KISSAT_BIN, str(Path(sys.executable).parent / "kissat"), shutil.which("kissat")):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError("no kissat binary: set MINIFLASH_KISSAT_BIN")


def _key(specification: dict, given_vals: Optional[list[dict]]) -> str:
    body = json.dumps({"spec": specification, "given": given_vals}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def solve(specification: dict, given_vals: Optional[list[dict]] = None, timeout: int = constant.LASSYNTH_TIMEOUT,
          cache: bool = True) -> Result:
    # kissat stops itself after timeout seconds, which lassynth reports as an unknown return code, the timeout here.
    from lassynth import LatticeSurgerySynthesizer
    depth = specification["max_k"] - 1
    path = constant.LASSYNTH_CACHE_DIR / f"{_key(specification, given_vals)}.json"
    if cache and path.exists():
        hit = json.loads(path.read_text())
        if hit["status"] != TIMEOUT or hit["timeout"] >= timeout:   # a cached timeout only covers budgets up to its own
            return Result(hit["status"], depth, hit["seconds"], hit["lasre"])

    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_dir:
        wrapper = Path(tmp_dir) / "kissat"
        wrapper.write_text(f'#!/bin/bash\nexec "{kissat_binary()}" --time={max(1, int(timeout))} "$@"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
        synthesizer = LatticeSurgerySynthesizer(solver="kissat", kissat_dir=tmp_dir)
        try:
            solution = synthesizer.solve(specification=specification, given_vals=given_vals)
            result = Result(UNSAT, depth, 0.0) if solution is None else Result(SAT, depth, 0.0, solution.lasre)
        except ValueError as error:
            if "return code" not in str(error):
                raise
            result = Result(TIMEOUT, depth, 0.0)
    result = Result(result.status, depth, time.perf_counter() - started, result.lasre)

    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps({"status": result.status, "seconds": result.seconds, "timeout": timeout,
                                        "lasre": result.lasre}, default=list))
        tmp_path.replace(path)
    return result


def check(window: Window, program: Program, problem: Problem, timeout: int = constant.LASSYNTH_TIMEOUT,
          **options) -> Result:
    # The window's spec at its own depth with the tile program's pipes given.
    return solve(spec(window, program, problem, **options), geometry(window, program), timeout)


def minimize(window: Window, program: Program, problem: Problem, timeout: int = constant.LASSYNTH_TIMEOUT,
             floor: int = 1, **options) -> list[Result]:
    # Results from floor layers upwards to the first satisfiable depth, a depth that ran out of time counting as unfound.
    out = []
    for depth in range(floor, layers(window) + 1):
        try:
            specification = spec(window, program, problem, depth=depth, **options)
        except ValueError:   # too shallow for the magic states of one tile
            continue
        out.append(solve(specification, timeout=timeout))
        if out[-1].status == SAT:
            break
    return out


def best(results: list[Result]) -> Optional[Result]:
    # The shallowest satisfiable result.
    sat = [result for result in results if result.status == SAT]
    return min(sat, key=lambda result: result.depth) if sat else None


def cell(window: Window, program: Program, problem: Problem, timeout: int = constant.LASSYNTH_TIMEOUT,
         **options) -> Cell:
    # When no depth below the window's own answers sat in time, the tile program's own pipes make the cell.
    results = minimize(window, program, problem, timeout, **options)
    found = best(results)
    below = [result for result in results if found is None or result.depth < found.depth]
    proven = found is not None and all(result.status == UNSAT for result in below)
    if found is not None and found.depth < layers(window):
        return Cell.of(window, found.lasre, proven)

    # Nothing shallower was found, and the tile program's own routing is the plainer solution.
    own = check(window, program, problem, max(timeout, constant.LASSYNTH_CHECK_TIMEOUT), **options)
    return Cell.of(window, own.lasre, proven)


def cells(slabs: Sequence[Slab], program: Program, problem: Problem, timeout: int = constant.LASSYNTH_TIMEOUT,
          report=None, **options) -> list[tuple[Slab, list[Cell]]]:
    # report(slab, cell) is called after each cell.
    out = []
    for slab in slabs:
        made = []
        for window in slab.windows:
            made.append(cell(window, program, problem, timeout, **options))
            if report is not None:
                report(slab, made[-1])
        out.append((slab, made))
    return out
