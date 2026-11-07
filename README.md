<p align="center">
  <img src="docs/logo.svg" width="128" height="128" alt="MiniFlash logo">
</p>

<h1 align="center">MiniFlash</h1>

<p align="center"><b>An <i>ultra fast</i> Clifford+T &rarr; lattice-surgery compiler.</b></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2626d9.svg" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-d92626.svg" alt="python 3.9+">
  <img src="https://img.shields.io/badge/pure-Python-26bf40.svg" alt="pure python">
</p>

MiniFlash turns an OpenQASM 2.0 circuit into a **tile program**: which tiles hold
qubits, magic states and routes, and which tiles merge at each step. It minimizes
spacetime volume (height × width × steps, walks included), checks every program
with an independent verifier, and renders it as a glTF scene.

## Quickstart

```bash
pip install -e '.[test]'
python -m pytest -q tests
python main.py benchmarks/algorithms/ghz8.qasm -o ghz8.gltf
```

Open the `.gltf` in any viewer, e.g. <https://gltf-viewer.donmccurdy.com/>. Time
runs upward, two layers per step. Qubits are red pillars; a CNOT's two halves sit
on the lower (blue) and upper (red) layer, T merges end at green magic states, and
walks between partitions are orange.

Several circuits print a CSV, one verified row each:

```bash
python main.py benchmarks/toffoli/*.qasm --solver placement > toffoli.csv
```

As a library:

```python
import miniflash as flash

problem = flash.read_qasm("benchmarks/toffoli/tof-3.qasm")
program = flash.make("layout").solve(problem)
report = flash.verify(problem, program)
flash.write_gltf(program, "tof3.gltf")
```

## Options

- `--solver`: a step of the compiler's ladder, `vanilla`, `dependency`, `routing`,
  `spacing`, `placement`, `layout` (default). `--shapes N` compares the N smallest
  site shapes.
- `--init`: the initial placement the anneal starts from: `spectral` (default),
  `random`, `gordian`, `gordian-l`, `mincut`, `aplace` or `mpl`.
- `--factory HxWxT`: magic-state factory, H tiles deep, W wide, one state every T
  steps, one behind every magic tile. `--factories F`: F shared factories feed the
  chip through delivery rings instead. Without either, magic supply is unlimited.
- `--json FILE`: save the tile program.

`benchmarks/` holds 206 circuits (GHZ, BV, DJ, QFT, random Clifford, graph states,
Galois-field multipliers, Toffoli, ...); see `benchmarks/README.md`.

## Citation

If you use MiniFlash in your research, please cite:

```bibtex
@article{wang2026miniflash,
  title   = {Demand-Driven Routing Space for Lattice-Surgery Compilation},
  author  = {Wang, Hanyu and Lin, Wan-Hsuan and Chen, Zihan and Zhang, Eddy Z. and Cong, Jason},
  journal = {arXiv preprint arXiv:2610.30747},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.30747}
}
```

## License

MIT
