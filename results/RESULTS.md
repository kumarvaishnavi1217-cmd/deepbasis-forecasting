# Results

All numbers are on the official M4 test set (100,000 series). Published reference numbers are copied from Tables 11-12 of Oreshkin et al. (2020). Rows marked *(ours)* are produced by DeepBasis; see `results/results.json` for the raw values.

## sMAPE (lower is better)

| Model | Yearly | Quarterly | Monthly | Others | Average |
|---|---:|---:|---:|---:|---:|
| N-BEATS-G *(published, 180 models)* | 13.023 | 9.212 | 12.048 | 3.574 | 11.168 |
| N-BEATS-I *(published, 180 models)* | 12.924 | 9.287 | 12.059 | 3.684 | 11.174 |
| N-BEATS-I+G *(published, 180 models)* | 12.913 | 9.213 | 12.024 | 3.643 | 11.135 |
| M4 winner (Smyl) *(published)* | 13.176 | 9.679 | 12.126 | 4.014 | 11.374 |
| **DeepBasis-G *(ours, 6-model ensemble)*** | 13.257 | 9.727 | 12.326 | 3.813 | 11.491 |
| Theta *(ours, statsforecast)* | 14.603 | 10.319 | 12.998 | 4.738 | 12.311 |
| SES *(ours, statsforecast)* | 16.381 | 11.081 | 14.371 | 6.775 | 13.664 |
| Naive2 *(ours; matches official M4 values exactly)* | 16.342 | 11.012 | 14.427 | 4.754 | 13.564 |

## OWA (lower is better; Naive2 = 1.000)

| Model | Yearly | Quarterly | Monthly | Others | Average |
|---|---:|---:|---:|---:|---:|
| N-BEATS-G *(published, 180 models)* | 0.765 | 0.800 | 0.820 | 0.822 | 0.797 |
| N-BEATS-I *(published, 180 models)* | 0.758 | 0.807 | 0.824 | 0.849 | 0.798 |
| N-BEATS-I+G *(published, 180 models)* | 0.758 | 0.800 | 0.819 | 0.840 | 0.795 |
| M4 winner (Smyl) *(published)* | 0.778 | 0.847 | 0.836 | 0.920 | 0.821 |
| **DeepBasis-G *(ours, 6-model ensemble)*** | 0.780 | 0.845 | 0.844 | 0.859 | 0.821 |
| Theta *(ours, statsforecast)* | 0.872 | 0.918 | 0.906 | 0.996 | 0.898 |
| SES *(ours, statsforecast)* | 1.002 | 1.020 | 1.039 | 1.332 | 1.028 |
| Naive2 *(ours; matches official M4 values exactly)* | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Accuracy vs. training budget (DeepBasis-G ensemble)

Each column uses the forecasts saved at that iteration by the *same* training runs.

| Group | 250 it | 500 it | 1000 it | 2000 it |
|---|---:|---:|---:|---:|
| Yearly sMAPE / OWA | 13.273 / 0.781 | 13.263 / 0.784 | 13.275 / 0.784 | 13.257 / 0.780 |
| Quarterly sMAPE / OWA | 9.871 / 0.867 | 9.856 / 0.861 | 9.870 / 0.859 | 9.727 / 0.845 |
| Monthly sMAPE / OWA | 12.704 / 0.880 | 12.524 / 0.865 | 12.508 / 0.857 | 12.326 / 0.844 |
| Others sMAPE / OWA | 4.048 / 0.899 | 3.881 / 0.868 | 3.813 / 0.859 | 3.813 / 0.859 |
| Average sMAPE / OWA | 11.722 / 0.840 | 11.622 / 0.834 | 11.616 / 0.831 | 11.491 / 0.821 |

Weekly/Daily/Hourly ("Others") are trained for 1000 iterations; later columns reuse their 1000-iteration forecasts.

## Ensembling gain and member breakdown (DeepBasis-G, final iteration)

| Group | mean single-model sMAPE | best single model | ensemble sMAPE |
|---|---:|---:|---:|
| Yearly | 13.834 | 13.543 (`generic-smape-L3-s0`) | 13.257 |
| Quarterly | 10.176 | 10.008 (`generic-smape-L3-s0`) | 9.727 |
| Monthly | 12.653 | 12.295 (`generic-smape-L3-s0`) | 12.326 |
| Others | 3.935 | 3.820 (`generic-smape-L5-s0`) | 3.813 |
| Average | 11.894 | 11.619 (`generic-smape-L3-s0`) | 11.491 |

### By training loss and lookback (sub-ensembles, Average sMAPE)

| Sub-ensemble | sMAPE | OWA |
|---|---:|---:|
| loss = smape | 11.534 | 0.825 |
| loss = mape | 11.821 | 0.841 |
| loss = mase | 11.660 | 0.833 |
| lookback = 3H | 11.547 | 0.828 |
| lookback = 5H | 11.769 | 0.840 |

## Interpretable model (single-model demo)

One DeepBasis-I model (trend + seasonality stacks, sMAPE loss, lookback 3H, 1000 iterations) on Monthly: **sMAPE 12.303, OWA 0.854**. That is on par with the 6-model DeepBasis-G ensemble on Monthly, while its forecast splits into readable trend and seasonality components (`results/figures/decomposition_monthly.png`).

