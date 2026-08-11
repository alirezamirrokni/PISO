# PISO Experimental Code

This repository contains the experimental code used to evaluate PISO and PISO² across the pricing, strategic-classification, bilevel-optimization, and performative-RL settings. The code is organized by experiment family, with shared output conventions and resumable caches for long runs.

## Repository structure

- `pricing & classification/` contains the pricing and strategic-classification experiments, together with all pricing ablations.
- `bilevel optimization/` contains the routing and security-game bilevel experiments.
- `performative rl/` contains the performative-RL gridworld experiments.
- `results/` is created automatically at runtime and is ignored by Git.

The former standalone pricing-ablation project has been merged into `pricing & classification/`. Pricing ablations now reuse the same pricing data and problem implementation as the main pricing experiment.

## Installation

Python 3.11 is recommended. From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

## Pricing and classification

The pricing, classification, and ablation experiments share one entry point:

```bash
python "pricing & classification/run.py" --problem pricing
python "pricing & classification/run.py" --problem classification
python "pricing & classification/run.py" --problem ablation
```

Default configurations are loaded automatically from:

- `pricing & classification/configs/pricing.yaml`
- `pricing & classification/configs/classification.yaml`
- `pricing & classification/configs/ablations.yaml`

The default outputs are written to `results/pricing/`, `results/classification/`, and `results/ablations/`, respectively. A different configuration or output location can be supplied with `--config` and `--output`.

### Pricing ablations

Available ablation suites are:

- `components`: known component, residual without momentum, PISO, and PISO².
- `gamma`: sweep over the known-gradient coefficient γ.
- `rho`: sweep over the residual-momentum coefficient ρ.
- `lambda`: sweep over the PISO² direct-residual mixing coefficient λ.
- `cycle_length`: sweep over the cycle length for the Cycle variants.

List the available suites with:

```bash
python "pricing & classification/run.py" --problem ablation --list-suites
```

Run one suite:

```bash
python "pricing & classification/run.py" --problem ablation --suite rho
```

Run several selected suites:

```bash
python "pricing & classification/run.py" --problem ablation --suite gamma --suite rho --suite lambda
```

If no `--suite` option is supplied, all ablation suites are run. The paper configuration uses 100 paired simulations. For a custom run, the number of simulations can be overridden without editing the configuration:

```bash
python "pricing & classification/run.py" --problem ablation --suite rho --simulations 20
```

Ablations support parallel execution with `--jobs N`. Completed runs are cached independently, so rerunning an experiment only computes missing or invalidated cache entries. `--plot-only` regenerates ablation reports from completed caches, while `--reset-cache` removes the selected experiment caches before execution.

Each ablation suite preserves the existing result layout. For example, the rho suite writes to `results/ablations/rho/` and produces its cache directory, trajectory tables, summary tables, `rho_ablation.png`, `rho_ablation.pdf`, and suite manifest. `results/ablations/all_suites_summary.csv` combines the summaries of the suites included in the invocation.

## Bilevel optimization

Run the routing or security experiment with:

```bash
python "bilevel optimization/run.py" --problem routing
python "bilevel optimization/run.py" --problem security
```

The default configurations are `bilevel optimization/configs/routing.yaml` and `bilevel optimization/configs/security.yaml`. Results are written to `results/bilevel/routing/` and `results/bilevel/security/`.

Useful options include:

```bash
python "bilevel optimization/run.py" --problem routing --jobs 8
python "bilevel optimization/run.py" --problem routing --methods GaussianPISO,CyclePISO,GaussianPISO2,CyclePISO2
python "bilevel optimization/run.py" --problem routing --plot-only
python "bilevel optimization/run.py" --problem routing --reset-cache
```

`--methods` accepts a comma-separated subset of methods configured for the selected experiment.

## Performative RL

Run the performative-RL experiment with:

```bash
python "performative rl/run_piso_rl.py"
```

The default configuration is `performative rl/configs/piso_rl.yaml`, and outputs are written to `results/piso_v4_heldout/`.

Common overrides include:

```bash
python "performative rl/run_piso_rl.py" --jobs 8
python "performative rl/run_piso_rl.py" --seeds 0,1,2
python "performative rl/run_piso_rl.py" --methods PISO,PISO2
python "performative rl/run_piso_rl.py" --reset-cache
```

To regenerate the performative-RL plots from existing result tables:

```bash
python "performative rl/plot_piso_rl.py"
```

## Data

The pricing data used by both the main pricing experiment and the pricing ablations are stored under `pricing & classification/data/prices/`. The processed credit dataset for strategic classification is stored under `pricing & classification/data/classification/`. The classification loader can also use the source URL specified in the configuration if the local file is absent and automatic download is enabled.

## Reproducibility and caching

Experiment seeds and method hyperparameters are defined in the YAML configuration files. Long-running experiments use resumable caches so interrupted runs can continue without recomputing completed work. Cache fingerprints include the numerical configuration and relevant implementation/data state; changing a numerical method, input data, or incompatible configuration causes the affected cache entries to be recomputed.

For final paper results, run the provided configurations without reducing sample budgets or simulation counts. Results, caches, generated figures, and generated tables are intentionally excluded from version control through the repository-level `.gitignore`.
