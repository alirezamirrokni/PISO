# PISO experiments: pricing and strategic classification

This project runs the complete pricing experiment and the strategic-classification experiment from Hikima et al. through one command-line interface.

It contains the five published baselines, the three PISO variants, and the three two-level PISO variants:

- `GZO_NS`, `GZO_HS`, `ZO_TG`, `ZO_OG`, `ZO_OGVR`
- `GaussianPISO`, `GuidedPISO`, `CyclePISO`
- `GaussianPISO2`, `GuidedPISO2`, `CyclePISO2`

For PISO²,

```text
m_(k+1) = rho * m_k + (1 - rho) * R_k
G_(k+1) = gamma * g_k + lambda * R_k + (1 - lambda) * m_(k+1)
```

## Install

```bash
python requirements.py
```

## Pricing

The former `reference.yaml` is now `configs/pricing.yaml`.

```bash
python run.py \
  --problem pricing \
  --config configs/pricing.yaml \
  --output results/pricing
```

Existing pricing caches remain reusable. Use the same output directory that contains the previous `cache/` folder and do **not** pass `--reset-cache`.

A direct compatibility audit is included:

```bash
python -m tools.verify_pricing_cache_compatibility
```

It checks the six problem fingerprints and all 66 pricing method fingerprints against the uploaded pre-classification project.

## Strategic classification

```bash
python run.py \
  --problem classification \
  --config configs/classification.yaml \
  --output results/classification
```

On the first classification run, the code downloads the processed credit dataset used by the reference implementation to:

```text
data/classification/credit_processed.csv
```

It can also be downloaded explicitly:

```bash
python -m tools.download_classification_data
```

The classification configuration uses:

- 11 standardized credit features and one intercept parameter;
- initial decision vector `x0 = 1` in 12 dimensions;
- strategic-response costs `tau = 0.50, 1.00, 2.00, 4.00, 8.00`;
- 20 simulations per value of `tau`;
- a 10,000-sample termination budget;
- training loss, test loss, test AUC, and test accuracy;
- mean trajectories with one-standard-error shadows and no interpolation.

The baseline settings come from Appendix A.2. `ZO_OGVR` keeps the paper's exact `d^(-1/2)` initial step rather than replacing it with a long decimal.

## Outputs

Each problem writes:

```text
summary.csv
final_scores.csv
figure_data.csv
figure_data_raw.csv
figure_data_selected_run.csv
figure.png
figure.pdf
config.yaml
cache/
```

Classification `summary.csv` reports mean, sample standard deviation, standard error, and run count for all four metrics at each `tau`.

## Validation

```bash
python -m unittest discover -s tests -v
python -m tools.verify_pricing_cache_compatibility
```

To rerun the family-level search on the downloaded official data:

```bash
python -m tools.tune_classification_piso \
  --official-data \
  --config configs/classification.yaml \
  --output tuning_official
```

See `CLASSIFICATION_TUNING.md` for the classification implementation and PISO search audit.
