# Analysis of 20261112T213204901721Z_285d174e

Populations: 960; settings that vary: phi, outside_option.
Main setting used below and in figures: {'rho': '0.5', 'phi': '1.0', 'outside_option': '-1.0', 'K': '10'}.

## Main-setting results (means; see summary.csv for 95% CIs)

recommender  cat_entropy  homogeneity  welfare_net  fatigue  rounds  dropout
       bm25        0.206        0.079        0.612    0.916   5.011    0.994
    content        0.226        0.086        0.567    1.044   5.694    0.980
      ideal        0.453        0.065        2.191    1.637  12.232    0.765
         mf        0.145        0.267        1.117    0.426   2.938    1.000
    mmr_0.6        0.344        0.063        0.518    1.272   7.570    0.956
       none        0.439        0.061        0.567    1.386   9.907    0.900
 popularity        0.138        0.334        0.876    0.434   2.993    1.000
     random        0.312        0.010        0.599    0.618   3.970    1.000

## Files

- summary.csv
- paired_vs_none.csv
- paired_vs_random.csv
- rank_robustness.csv
- figures/coverage_vs_utility.png
- figures/bubble_vs_echo_chamber.png

Paired differences compare each recommender with the reference on the same simulated
readers (matched by seed); a difference is marked significant when its bootstrap
95% CI excludes zero. Recommenders marked reference-only (ideal) use readers' true
preferences and are excluded from rank robustness.
