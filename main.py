# Compile circuits into verified tile programs and print one CSV row each. A .qasm file compiles
# as a flat DAG, a hierarchy .json bottom-up through its call tree. The knobs are in
# miniflash/constant.py; the glTF scene of each circuit lands in the output directory under the
# circuit's name, or at the given .gltf path when there is one circuit.
import argparse, csv, sys, time
from pathlib import Path

import miniflash as flash
from miniflash import constant

parser = argparse.ArgumentParser(description="compile circuits to lattice-surgery tile programs")
parser.add_argument("circuits", nargs="+", help="qasm circuits or hierarchy json files")
parser.add_argument("-o", "--output", type=Path, default=constant.OUTPUT_DIR, help="output directory, or a .gltf file for one circuit")
args = parser.parse_args()

single = args.output.suffix.lower() == ".gltf"
out_dir = args.output.parent if single else args.output

FIELDS = ["circuit", "qubits", "pairs", "depth", "grid", "steps", "walk_steps", "volume", "compile_s", "verify_s"]

factory = flash.Factory() if constant.LIMITED_SUPPLY else None
factories = constant.SHARED_FACTORIES if constant.LIMITED_SUPPLY else 0
solver = flash.make(constant.SOLVER, shapes=constant.LAYOUT_SHAPES, factory=factory, factories=factories)
out_dir.mkdir(parents=True, exist_ok=True)

writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
writer.writeheader()
for path in args.circuits:
    name = Path(path).stem
    hierarchical = Path(path).suffix.lower() == ".json"

    started = time.perf_counter()
    if hierarchical:
        hierarchy = flash.Hierarchy.load(path)
        problem = hierarchy.problem
        program = solver.solve_hierarchy(hierarchy, constant.HIERARCHY_PLACEMENT)
    else:
        problem = flash.read_qasm(path)
        program = solver.solve(problem)
    compiled = time.perf_counter()
    report = flash.verify(problem, program, factory, factories)
    verified = time.perf_counter()

    writer.writerow({"circuit": name, "qubits": problem.n_qubits, "pairs": len(problem.pairs), "depth": problem.depth,
                "grid": f"{report.height}x{report.width}", "steps": report.steps, "walk_steps": report.walk_steps, "volume": report.volume,
                "compile_s": round(compiled - started, 3), "verify_s": round(verified - compiled, 3)})
    sys.stdout.flush()
    print(f"{solver.name} {name}: {report.steps} steps, volume {report.volume}, {compiled - started:.2f}s", file=sys.stderr, flush=True)
    if hierarchical:
        stats = solver.hierarchy_stats
        print(f"hierarchy: {stats['calls']} calls, {stats['cells']} distinct cells, "
              f"{stats['routed_pairs']}/{stats['total_pairs']} pairs routed", file=sys.stderr, flush=True)

    if constant.WRITE_PROGRAM:
        program.save(out_dir / f"{name}.json")
    if not constant.WRITE_GLTF:
        continue
    out = args.output if single else out_dir / f"{name}.gltf"

    supply = None
    if factories:
        supply = flash.plan(program, factories, factory)
        for conflict in supply.conflicts:
            print(f"supply conflict: {conflict}", file=sys.stderr)

    # the resynthesis windows over the program, carved out as boxes, or replaced by the cells lassynth finds
    overlays = ()
    carve = False
    if constant.RENDER_WINDOWS:
        from miniflash.resynthesis import boxes, extract_dp, extract_greedy, write_cells
        if constant.WINDOW_CUT == "dp":
            picked = extract_dp(program, problem)
        else:
            picked = extract_greedy(program, problem, min_gain=constant.WINDOW_MIN_GAIN)
        carve = constant.RENDER_WINDOWS == "coarse"
        overlays = boxes(picked, alpha=1.0 if carve else constant.WINDOW_ALPHA)
        print(f"windows: {sum(len(slab.windows) for slab in picked)} in {len(picked)} slabs covering "
              f"{sum(slab.steps for slab in picked)} of {report.steps} steps", file=sys.stderr)
    if constant.RENDER_WINDOWS in ("cells", "cells-inplace"):
        from miniflash.resynthesis.synth import cells

        def announce(slab, cell):
            print(f"cell {slab.start}-{slab.stop} {cell.window.box.height}x{cell.window.box.width}: "
                  f"{constant.LAYERS_PER_STEP * cell.window.steps} to {cell.depth} layers", file=sys.stderr, flush=True)

        plan = cells(picked, program, problem, timeout=constant.LASSYNTH_TIMEOUT, report=announce)
        layers = write_cells(program, plan, out, compact=constant.RENDER_WINDOWS == "cells")
        print(f"cells: {constant.LAYERS_PER_STEP * report.steps} to {layers} layers, volume "
              f"{constant.LAYERS_PER_STEP * report.volume} to {layers * report.height * report.width} tile-layers", file=sys.stderr)
    else:
        flash.write_gltf(program, out, supply, overlays=overlays, carve=carve)
