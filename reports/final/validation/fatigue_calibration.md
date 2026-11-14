# Fatigue estimated from MIND click histories

**Data.** 20,000 users with at least 10 clicks.

## 1. Streaks versus a shuffled baseline

Shuffling each user's history keeps their topic mix but removes order. Continuation
below the shuffled value indicates satiation, above it indicates momentum.

streak_k  transitions  p_continue_observed  p_continue_shuffled  ratio_observed_to_shuffled
       1       568312               0.0722               0.0672                      1.0738
       2        39787               0.1578               0.1569                      1.0060
       3         6099               0.2459               0.2402                      1.0237
       4         1451               0.3170               0.3190                      0.9939
       5          445               0.3888               0.3794                      1.0246
      6+          353               0.5552               0.5107                      1.0871

After k = 2 same-subcategory clicks, continuation is
1.01x the shuffled value, i.e. above the shuffled baseline (momentum: readers binge a topic before moving on).
The raw switch rate after one click mostly reflects how many subcategories exist
and how spread out each user's interests are, not fatigue.

## 2. Choice model without look-ahead

Baseline preferences come from the first half of each history; clicks in the second
half are predicted. Decay chosen by held-out log-likelihood (70/30 user split).

 decay  heldout_loglik  b_pref  b_pref_se  b_fatigue_observed  b_fatigue_observed_se  b_fatigue_shuffled  b_fatigue_shuffled_se  order_effect  order_effect_se  order_effect_odds_ratio  train_events  test_events
  0.30         -0.6455  6.3433     0.0554              0.7783                 0.0170              0.6731                 0.0167        0.1052           0.0238                   1.1110        248990       103866
  0.50         -0.6438  5.9857     0.0565              0.8210                 0.0153              0.7205                 0.0151        0.1004           0.0215                   1.1056        248990       103866
  0.60         -0.6427  5.6975     0.0576              0.8191                 0.0141              0.7263                 0.0139        0.0928           0.0197                   1.0972        248990       103866
  0.70         -0.6410  5.2507     0.0593              0.8006                 0.0125              0.7188                 0.0123        0.0818           0.0176                   1.0853        248990       103866
  0.80         -0.6380  4.4326     0.0627              0.7729                 0.0105              0.7057                 0.0104        0.0672           0.0148                   1.0695        248990       103866
  0.90         -0.6294  2.4770     0.0694              0.7479                 0.0080              0.7015                 0.0079        0.0464           0.0112                   1.0475        248990       103866
  0.95         -0.6190  1.0576     0.0700              0.6445                 0.0057              0.6148                 0.0056        0.0297           0.0080                   1.0301        248990       103866

**Best decay: 0.95.** Order effect = 0.030
(SE 0.008, z = 3.7): beyond what a user's topic mix
explains, one recent read in a subcategory multiplies the odds of choosing it again by
1.030. This is momentum (repeat reading).

## 3. Use in the simulator

- The decay estimate (0.95) can inform `fatigue_decay`.
- The order effect is on a logit scale, not the simulator's utility scale, so it does
  not give phi directly. Its sign says whether fatigue (or momentum) is present in real
  reading; the phi values used (and the reader-type values) are modelling assumptions,
  and results should be reported across a range of phi, including phi = 0.
- Caveats: MIND histories are click logs from a live recommender, so order also
  reflects what was shown; subcategory is a coarse proxy for similarity.
