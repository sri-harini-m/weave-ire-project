# Analysis of 20261113T060434478550Z_d9f44e99

Populations: 2160; settings that vary: gamma, beta, sigma_bar.
Main setting used below and in figures: {'gamma': '1.0', 'beta': '1.0', 'sigma_bar': '1.0', 'K': '10'}.

## Main-setting results (means; see summary.csv for 95% CIs)

recommender  cat_entropy  homogeneity  welfare_net  fatigue  rounds  dropout
       bm25        0.308        0.207        0.018    1.107    20.0      0.0
    content        0.314        0.187        0.045    1.203    20.0      0.0
      ideal        0.530        0.093        2.371    1.039    20.0      0.0
         mf        0.728        0.364        0.025    0.955    20.0      0.0
    mmr_0.6        0.449        0.093        0.540    1.119    20.0      0.0
       none        0.507        0.090        0.782    1.041    20.0      0.0
 popularity        0.542        0.685       -0.471    0.828    20.0      0.0
     random        0.584        0.043        0.156    0.928    20.0      0.0

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
