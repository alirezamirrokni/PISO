# PISO and PePG Performative-RL Experiment

Final experiment-only project for the PePG gridworld. It contains the tuned
`configs/piso_rl.yaml`, six PISO/PISO² methods, trajectory-based baselines,
PePG, resumable caches, and mean ± standard-error reports.

## CPU optimizations

The experiment remains CPU-based, but the expensive path has been optimized:

- rollout deployment no longer constructs exact reward/transition/occupancy tensors;
- exact metrics are evaluated every 1,000 trajectories and at the final iterate;
- identical deployed policies are cached and reused;
- policy probabilities are computed once per known-gradient batch;
- gridworld follower response, exact transitions, and occupancy are vectorized;
- PePG evaluates learned reward/transition models once per rollout batch;
- independent method/seed runs use two worker processes by default.

The optimized rollout and exact-evaluation calculations were checked against the
previous implementation and preserve its trajectories, gradients, and final
iterates up to floating-point roundoff.

## Included methods

- `VanillaPG`
- `ZO_TG`, `ZO_OG`
- `GZO_NS`, `GZO_HS`
- `GaussianPISO`, `GuidedPISO`, `CyclePISO`
- `GaussianPISO2`, `GuidedPISO2`, `CyclePISO2`
- `PePG`

## Install in Colab

```bash
python -m pip install -q -r requirements_colab.txt
```

Disable online W&B logging:

```python
import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["MPLBACKEND"] = "Agg"
```

## Run

Use a Google Drive output directory so caches survive disconnects:

```bash
python -u run_piso_rl.py \
  --config configs/piso_rl.yaml \
  --output /content/drive/MyDrive/PISO_performative_RL/final_complete_experiment
```

The config uses 20 seeds, 100,000 trajectories per method/seed,
`evaluation_interval: 1000`, and `n_jobs: 2`.

Override the number of workers when needed:

```bash
python -u run_piso_rl.py --output "$OUTPUT" --jobs 2
```

For safer Colab execution, run method groups with the same output directory:

```bash
python -u run_piso_rl.py --output "$OUTPUT" \
  --methods VanillaPG,ZO_TG,ZO_OG,GZO_NS,GZO_HS

python -u run_piso_rl.py --output "$OUTPUT" \
  --methods GaussianPISO,GuidedPISO,CyclePISO

python -u run_piso_rl.py --output "$OUTPUT" \
  --methods GaussianPISO2,GuidedPISO2,CyclePISO2

python -u run_piso_rl.py --output "$OUTPUT" --methods PePG
```

Rerun the same command after a disconnect. Completed runs load from `final.pkl`;
incomplete runs resume from `progress.pkl`. Do not use `--reset-cache` unless you
intend to discard the corresponding runs.

After every group finishes, run the full command once to create the combined
report for all methods.

## Outputs

- `cache/`: per-method/per-seed resumable caches
- `raw_runs.csv`
- `figure_data.csv`
- `final_scores.csv`
- `summary.csv`
- `figure.png` and `figure.pdf`
- `config.yaml`
