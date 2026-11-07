# Compare flat compilation and cell reuse on the same Qualtran inputs. Every row passes the
# verifier; solve_s excludes JSON loading and verification, load_s is the hierarchy expansion.
# Run with OPENBLAS_NUM_THREADS=1 for reproducible timing.
import argparse, csv, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from miniflash import Hierarchy, make, verify

CIRCUITS = ["modexp-3", "modmul-4", "gf2inv-4", "modadd-8", "add-16", "hubbard-2", "qrom-6", "gf2mul-8"]
FIELDS = ["circuit", "seed", "mode", "pairs", "cells", "calls", "leaf_cells", "composed_cells", "leaf_calls", "routed_pairs", "volume", "steps", "walks",
          "load_s", "solve_s", "verify_s", "extract_s", "map_s", "route_s", "compose_s", "windows_s", "compact_s"]

parser = argparse.ArgumentParser(description="flat compilation against cell reuse")
parser.add_argument("circuits", nargs="*", help="benchmark stems or json paths")
parser.add_argument("--seeds", type=int, nargs="+", default=[0])
parser.add_argument("--modes", nargs="+", choices=["flat", "weighted", "hierarchy-flat"], default=["flat", "weighted", "hierarchy-flat"])
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for name in args.circuits or CIRCUITS:
        path = Path(name) if name.endswith(".json") else ROOT / "benchmarks/qualtran" / f"{name}.json"
        started = time.perf_counter()
        hierarchy = Hierarchy.load(path)
        load_s = time.perf_counter() - started

        for seed in args.seeds:
            for mode in args.modes:
                compiler = make("layout", seed=seed)
                started = time.perf_counter()
                if mode == "flat":
                    program = compiler.solve(hierarchy.problem)
                else:
                    program = compiler.solve_hierarchy(hierarchy, "weighted" if mode == "weighted" else "flat")
                solved = time.perf_counter()
                report = verify(hierarchy.problem, program)
                checked = time.perf_counter()

                stats = compiler.hierarchy_stats
                profile = compiler.profile[0]   # the stats layout compiles one predicted shape
                row = dict(circuit=path.stem, seed=seed, mode=mode, pairs=len(hierarchy.problem.pairs),
                           cells=stats.get("cells", 0), calls=stats.get("calls", 0),
                           **{key: stats.get(key, 0) for key in ("leaf_cells", "composed_cells", "leaf_calls")},
                           routed_pairs=stats.get("routed_pairs", len(hierarchy.problem.pairs)),
                           volume=report.volume, steps=report.steps, walks=report.walk_steps,
                           load_s=load_s, solve_s=solved - started, verify_s=checked - solved,
                           extract_s=stats.get("extract_s", 0),
                           **{key: profile[key] for key in ("map_s", "route_s", "compose_s", "windows_s", "compact_s")})
                writer.writerow(row)
                out.flush()
                print(f"{path.stem} seed={seed} {mode}: {solved - started:.3f}s, volume={report.volume}", flush=True)
