# Validation record

The `reference.yaml` profile was checked with seed 2024, 20 random instances per week, a 5,000-sample stopping threshold, and 1,000 fresh samples per metric evaluation. The aggregate objective means and sample standard deviations agree with the six published rows after rounding to two decimal places.

The exact checked values are stored in `data/benchmark_reference.csv`. The first weekly row was also executed through the packaged CLI, and the complete test suite passed.

Reference-mode terminal sample counts are:

| Method | Final counted samples |
|---|---:|
| GZO_NS | 5046 |
| GZO_HS | 5092 |
| ZO_TG | 5092 |
| ZO_OG | 5046 |
| ZO_OGVR | 5066 |

Counts exceed 5,000 because termination is checked after completing an iteration.
