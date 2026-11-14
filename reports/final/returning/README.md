# Analysis of 20261112T201046912499Z_82f98db7

Populations: 840; settings that vary: phi, reader_history_visible.
Main setting used below and in figures: {'rho': '0.5', 'phi': '0.5', 'K': '10', 'reader_history_visible': 'False'}.

## Main-setting results (means; see summary.csv for 95% CIs)

recommender  cat_entropy  homogeneity  welfare_net  fatigue  rounds  dropout
       bm25        0.310        0.200        0.016    1.105    20.0      0.0
 calibrated        0.209        0.211        0.048    1.191    20.0      0.0
    content        0.320        0.185        0.064    1.203    20.0      0.0
         mf        0.718        0.368        0.046    0.951    20.0      0.0
    mmr_0.6        0.453        0.091        0.551    1.119    20.0      0.0
       none        0.509        0.089        0.786    1.040    20.0      0.0
 popularity        0.540        0.688       -0.512    0.827    20.0      0.0

## Files

- summary.csv
- paired_vs_none.csv
- paired_returning_vs_new.csv
- rank_robustness.csv
- figures/coverage_vs_utility.png
- figures/bubble_vs_echo_chamber.png

Paired differences compare each recommender with the reference on the same simulated
readers (matched by seed); a difference is marked significant when its bootstrap
95% CI excludes zero. Recommenders marked reference-only (ideal) use readers' true
preferences and are excluded from rank robustness.
