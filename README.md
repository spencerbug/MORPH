# MORPH research super-repository

Research based on [MORPH: Model of Object Representation and Predictive Homomorphisms](https://spencerbug.github.io/blog/morph/), growing through small experiments before larger systems and demonstrations.

| Location | Role |
|---|---|
| [architecture/](architecture/capture.md) | Intended architecture, source interpretation and open questions |
| [experiments/e0_rotation/](experiments/e0_rotation/README.md) | Self-contained controlled rotation experiment: code, config, tests, reports |
| [packages/](packages/README.md) | Shared code only when multiple consumers justify extraction |
| [demos/](demos/README.md) | Future demonstrations with explicit experiment/checkpoint provenance |
| `artifacts/<experiment>/` | Generated datasets, checkpoints, predictions and run provenance; ignored by Git |
| [SPEC.md](SPEC.md) | Current active experiment contract |
| [Experiments.md](Experiments.md) | Append-only research decisions, outcomes and failures |
| [STATUS.md](STATUS.md) | Current state and exact resume commands |
| [AGENTS.md](AGENTS.md) | Automated research operating constraints |

All development, environments and caches stay in this project. Research docs and compact reports are versionable; generated artifacts stay separate. Future experiments get new directories and immutable specifications rather than changing old experiments in place.

Environment recreation, using an installed `uv` executable:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.tools/python" uv python install 3.12
UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.tools/python" uv venv --python 3.12 .venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python -r requirements-lock.txt
```

Latest result: [E0 report](experiments/e0_rotation/reports/REPORT.md). Structured prediction passed the registered gates, reducing eight-step error by 54.8%; unrestricted-baseline optimization limits the mechanistic conclusion. No later mechanism or demonstration has been implemented.
