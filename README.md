# Decision-Dependent Pricing Benchmark

This package implements a stochastic multiproduct pricing benchmark with five zeroth-order optimization methods. The implementation is organized around interchangeable method classes, YAML configuration, deterministic random streams, raw trajectory logging, statistical summaries, and publication-style reporting.

## Environment

The target interpreter is Python 3.12.2. Create an isolated environment and install the package:

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Main run

```bash
python -m ddp_pricing.cli run \
  --config configs/reference.yaml \
  --output outputs/reference
```

The command writes:

- `summary.csv`: mean, sample standard deviation, and paired-test values.
- `summary.tex`: LaTeX table.
- `trajectories.csv`: every recorded objective value and sample count.
- `final_runs.csv`: one row per week, instance, and method.
- `figure.png` and `figure.pdf`: six-panel objective trajectories.
- `resolved_config.yaml`: exact configuration used.
- `environment.json`: interpreter and package versions.

The expected full-run statistics are stored in `data/benchmark_reference.csv`. Compare a completed run with:

```bash
python scripts/compare_summary.py outputs/reference/summary.csv
```

The full run contains 20 random problem instances for each of six weekly datasets. A fast integration check is available with:

```bash
python -m ddp_pricing.cli run --config configs/quick.yaml --output outputs/quick
```

## Adding a method

1. Add a class under `src/ddp_pricing/methods/` that implements:

```python
class MyMethod:
    name = "MY_METHOD"

    def __init__(self, params: dict):
        ...

    def run(self, problem, rng, run_context):
        ...
        return trace
```

2. Add its import path to the `methods` list in a YAML file:

```yaml
methods:
  - name: MY_METHOD
    factory: ddp_pricing.methods.my_method:MyMethod
    params:
      step_size: 0.01
```

No runner or reporting code must be modified.

## Configuration profiles

- `configs/reference.yaml` follows the released implementation behavior, including its random-number order, sample accounting, update order, and numerical constants.
- `configs/appendix.yaml` uses the numerical constants and covariance scaling printed in the manuscript appendix.
- `configs/quick.yaml` is a low-cost smoke configuration.

The two full profiles are intentionally separate because the released implementation and the printed appendix differ in several operational details. The selected profile is copied into the output directory.

## Data

The price files contain the public weekly confectionery price observations used by the benchmark. The optimization uses the first ten entries of each file, then divides them by their maximum. Source and provenance details are in `THIRD_PARTY_NOTICES.md`.
