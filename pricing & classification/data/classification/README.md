# Strategic-classification dataset

The classification runner expects the processed credit-card default dataset used by the reference implementation:

```text
data/classification/credit_processed.csv
```

The default configuration downloads it from the Actionable Recourse repository automatically. To download it before a run:

```bash
python -m tools.download_classification_data
```

Preprocessing follows the reference experiment:

1. remove the six marriage/age indicator columns;
2. standardize the remaining 11 features;
3. balance the two labels;
4. deterministically shuffle with `dataset_seed: 0`;
5. use 12,272 training examples and 1,000 test examples.
