# Validation of the reader choice rule on real MIND dev impressions

**Setup.** 10,000 real impressions from `data/raw/dev/behaviors.tsv` (users with at least 5
earlier clicks; the 30 most recent clicks are treated as already read).
Each shown article is scored with the simulator's choice rule,
mean - gamma/2 x variance - phi x fatigue, using the simulator's kernels
(rho = 0.5, fatigue_rho = 0.8, 8-dim PCA,
prior variance 2). Past reads are assumed to carry no surprise, so they change
variance and fatigue but not the mean. 95% CIs are normal approximations over impressions.
Users are split in two halves by user id: (gamma, phi) is chosen on the fit half
(4,926 impressions) and every row below is scored on the held-out test half
(5,074 impressions).

**Held-out test half**

| Scorer | ROC-AUC | Top-1 hit rate | MRR |
|---|---|---|---|
| random | 0.509 [0.501, 0.517] | 10.5% [9.7%, 11.4%] | 0.249 [0.241, 0.257] |
| popularity | 0.527 [0.520, 0.533] | 11.0% [10.2%, 11.8%] | 0.260 [0.252, 0.268] |
| prior similarity (gamma=0, phi=0) | 0.645 [0.637, 0.653] | 19.8% [18.7%, 20.9%] | 0.362 [0.353, 0.372] |
| type: cautious | 0.647 [0.639, 0.655] | 20.1% [19.0%, 21.2%] | 0.364 [0.355, 0.374] |
| type: explorer | 0.644 [0.636, 0.652] | 19.6% [18.6%, 20.7%] | 0.360 [0.351, 0.370] |
| type: easily_bored | 0.640 [0.631, 0.648] | 19.2% [18.2%, 20.3%] | 0.357 [0.347, 0.367] |
| type: average | 0.645 [0.637, 0.653] | 19.8% [18.7%, 20.9%] | 0.362 [0.353, 0.372] |
| best fit on fit half (gamma=4, phi=0.2) | 0.650 [0.642, 0.658] | 19.7% [18.6%, 20.8%] | 0.365 [0.355, 0.374] |

Gain of the best fit over plain similarity on the test half (paired, same impressions):
AUC +0.0048 [+0.0022, +0.0075].

**AUC over the (gamma, phi) grid, fit half** (rows gamma, columns phi; fatigue decay 0.8):

|   gamma |    0.0 |    0.2 |    0.5 |    1.0 |    1.5 |
|--------:|-------:|-------:|-------:|-------:|-------:|
|     0   | 0.6521 | 0.6511 | 0.6496 | 0.6448 | 0.6395 |
|     0.5 | 0.6527 | 0.6522 | 0.6506 | 0.6466 | 0.6416 |
|     1   | 0.653  | 0.6524 | 0.6516 | 0.6478 | 0.6443 |
|     2   | 0.6549 | 0.6541 | 0.6528 | 0.6511 | 0.6466 |
|     4   | 0.6559 | 0.6561 | 0.655  | 0.6542 | 0.6508 |
|     8   | 0.6527 | 0.6529 | 0.654  | 0.6555 | 0.6554 |

Best fit on the fit half: gamma = 4, phi = 0.2 (fit-half AUC 0.6561).
gamma = phi = 0 is the plain similarity ranking used in the earlier check.
