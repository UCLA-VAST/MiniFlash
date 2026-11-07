"""Compare initial placements under the full flow (the `layout` solver): each seed
alone, and each seed refined by the default anneal.

    OPENBLAS_NUM_THREADS=1 python scripts/benchmark_init.py tof-3.qasm --inits spectral gordian \\
        --seeds 0 1 2 --anneal both --output out/init.csv

Each row passes the independent verifier. `initial_cost` and `final_cost` are the
placement proxy before and after the anneal; `init_s` times the seed alone (rerun
once after solving), `map_s` the seed plus the anneal.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from miniflash import INITS, make, read_qasm, verify
from miniflash.synthesis.placement import initial_order

FIELDS = ["circuit", "qubits", "pairs", "init", "anneal", "iters", "seed", "shape", "grid", "steps", "walk_steps", "volume",
          "initial_cost", "final_cost", "init_s", "map_s", "compile_s", "verify_s"]


def run(path: Path, init: str, anneal: bool, seed: int, per_qubit: int = 100) -> dict:
    problem = read_qasm(path)
    compiler = make("layout", init=init, anneal=anneal, seed=seed, iters=per_qubit * problem.n_qubits)
    started = time.perf_counter()
    program = compiler.solve(problem)
    compile_s = time.perf_counter() - started
    report = verify(problem, program)
    verify_s = time.perf_counter() - started - compile_s
    rows, cols = compiler.shape
    started = time.perf_counter()
    initial_order(init, problem, rows, cols, None, seed)
    init_s = time.perf_counter() - started
    return {"circuit": path.stem, "qubits": problem.n_qubits, "pairs": len(problem.pairs), "init": init,
            "anneal": int(anneal), "iters": per_qubit if anneal else 0, "seed": seed,
            "shape": f"{rows}x{cols}", "grid": f"{report.height}x{report.width}",
            "steps": report.steps, "walk_steps": report.walk_steps, "volume": report.volume,
            "initial_cost": round(compiler.initial_cost, 2), "final_cost": round(compiler.final_cost, 2),
            "init_s": round(init_s, 4), "map_s": round(sum(p["map_s"] for p in compiler.profile), 4),
            "compile_s": round(compile_s, 3), "verify_s": round(verify_s, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("circuits", nargs="+", type=Path, help=".qasm files")
    ap.add_argument("--inits", nargs="+", choices=INITS, default=list(INITS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--anneal", choices=["on", "off", "both"], default="both")
    ap.add_argument("--iters", type=int, default=100, help="anneal proposals per qubit (default 100)")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    anneals = {"on": [True], "off": [False], "both": [False, True]}[a.anneal]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for path in a.circuits:
            for init in a.inits:
                for anneal in anneals:
                    for seed in a.seeds:
                        row = run(path, init, anneal, seed, a.iters)
                        writer.writerow(row)
                        out.flush()
                        print(f"{row['circuit']} {init} anneal={int(anneal)} seed={seed}: volume {row['volume']}, "
                              f"{row['compile_s']}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
