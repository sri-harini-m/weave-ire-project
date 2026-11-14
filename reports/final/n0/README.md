# Analysis of 20261112T215239856851Z_af47a048

Populations: 480; settings that vary: uncertainty_n0.
Main setting used below and in figures: {'rho': '0.5', 'phi': '0.5', 'uncertainty_n0': '20', 'K': '10'}.

## Main-setting results (means; see summary.csv for 95% CIs)

recommender  cat_entropy  homogeneity  welfare_net  fatigue  rounds  dropout
       bm25        0.313        0.199        0.196    1.103    20.0      0.0
    content        0.329        0.186        0.252    1.199    20.0      0.0
      ideal        0.556        0.093        2.451    1.030    20.0      0.0
         mf        0.725        0.384        0.111    0.948    20.0      0.0
    mmr_0.6        0.468        0.096        0.705    1.114    20.0      0.0
       none        0.523        0.093        0.920    1.028    20.0      0.0
 popularity        0.541        0.694       -0.407    0.827    20.0      0.0
     random        0.591        0.044        0.294    0.925    20.0      0.0

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
