# Implementation notes

DeepBasis uses the N-BEATS architecture. This file records the design decisions: what is taken directly from the original paper, the details the paper leaves open and how DeepBasis settles them, and where the training budget was scaled down.

Architecture reference: B. N. Oreshkin, D. Carpov, N. Chapados, Y. Bengio. *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.* ICLR 2020. [arXiv:1905.10437](https://arxiv.org/abs/1905.10437)

## Taken from the original architecture paper

| Item | Paper | This repo |
|---|---|---|
| Data | M4, 100k series, official train/test split | same (raw CSVs from `Mcompetitions/M4-methods`) |
| Horizons H | 6/8/18/13/14/48 | same |
| Generic architecture | 30 stacks × 1 block, 4 FC layers, width 512, no sharing (Table 18) | same (`model.generic`) |
| Interpretable architecture | Trend stack with 3 blocks, width 256, polynomial degree 2. Seasonality stack with 3 blocks, width 2048, Fourier harmonics i = 0…⌊H/2−1⌋. Sharing at stack level | same (`model.interpretable`) |
| Doubly residual stacking | backcast subtracted, forecasts summed | same, with the input reversed and the last value used as the initial level |
| Lookbacks | 2H…7H | 2H…7H available. The laptop run uses {3H, 5H} |
| Training-window sampling | Anchor drawn from the last L_H·H points. L_H = 1.5 (Y/Q/M) or 10 (W/D/H). Missing points zero-filled and masked | same (`WindowSampler`), anchor drawn uniformly |
| Batch | 1024 series sampled uniformly with replacement | same |
| Optimiser | Adam, initial lr 1e-3 | same |
| Training losses | sMAPE, MAPE, MASE (ensemble across all three) | same |
| Ensemble aggregation | median | same |
| Metrics | sMAPE, MASE, OWA relative to Naive2 | same. Validated: our Naive2 reproduces the official M4 Naive2 sMAPE and MASE **to three decimals for all six frequencies** |

## Design decisions (not specified in the paper)

- **Learning-rate schedule.** The paper says only "initial learning rate 0.001". We halve the learning rate at 1/3 and 2/3 of training. Set `lr_decay=False` in `RunConfig` to turn this off.
- **Gradient clipping.** We clip the global norm to 1.0 for stability. This isn't mentioned in the paper.
- **MASE scale.** The printed formula sums the seasonal-naive scale up to T+H. Evaluation follows the official M4 definition, which uses the in-sample part only. This is the definition that reproduces the published Naive2 numbers. The MASE *training* loss computes its scale over the masked lookback window of each sample.
- **sMAPE training loss.** The denominator is computed from detached values (`|ŷ|.detach() + |y|`), with zero-division mapped to 0.
- **"Others" group.** Weekly, Daily and Hourly are pooled (5,000 series), as in the M4 competition. OWA is computed from pooled sMAPE and MASE, never from averaged per-series ratios.
- **Composition of the 180-model ensemble.** The paper doesn't spell it out. 3 losses × 6 lookbacks × 10 random initialisations = 180 is the natural reading and is what `--profile paper` runs.

## Training budget

The full protocol trains 180 models of about 25M parameters each, for 15k iterations (Y/Q/M) or 5k iterations (W/D/H), per frequency and per architecture. That comes to **2,160 training runs**. On the Apple M1 (8 GB) used here, one 15k-iteration run takes about 55 minutes (about 0.2 s per iteration on the MPS GPU), so the full protocol would take about 80 days.

The `laptop` profile (`deepbasis/cli.py`) keeps the architecture, sampling, losses and evaluation unchanged and shrinks only the budget:

| | Paper | Laptop profile | Factor |
|---|---|---|---|
| Ensemble members per frequency | 180 | 6 (3 losses × lookbacks {3H, 5H} × 1 seed) | 1/30 |
| Iterations, Y/Q/M | 15,000 | 2,000 | 1/7.5 |
| Iterations, W/D/H | 5,000 | 1,000 | 1/5 |
| Architectures | G and I (and I+G) | G (the I architecture is shown only as a single-model interpretability demo) | |
| Total training | about 2,160 runs | 36 runs, about 4 h | |

Expect our numbers to be somewhat *worse* than the paper's because of this. The question DeepBasis answers is: **how much of the published accuracy do you keep at 0.5% of the compute?** (54k model-iterations here, against 10.8M for the published N-BEATS-G ensemble) The budget-curve table in `results/RESULTS.md` makes that trade-off explicit. `make train-paper` runs the full protocol on suitable hardware with no code changes.

## Validation of the evaluation pipeline

| Frequency | Naive2 sMAPE (ours / official) | Naive2 MASE (ours / official) | Theta sMAPE (ours via statsforecast / official M4 benchmark) |
|---|---|---|---|
| Yearly | 16.342 / 16.342 | 3.974 / 3.974 | 14.603 / 14.593 |
| Quarterly | 11.012 / 11.012 | 1.371 / 1.371 | 10.319 / 10.311 |
| Monthly | 14.427 / 14.427 | 1.063 / 1.063 | 12.998 / 13.002 |
| Weekly | 9.161 / 9.161 | 2.777 / 2.777 | 9.143 / 9.093 |
| Daily | 3.045 / 3.045 | 3.278 / 3.278 | 3.052 / 3.053 |
| Hourly | 18.383 / 18.383 | 2.395 / 2.395 | 18.133 / 18.138 |

The official values come from the M4 competition benchmark results. Small Theta differences come from statsforecast's implementation and are expected.
