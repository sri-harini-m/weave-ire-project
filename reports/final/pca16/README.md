# Analysis of 20261113T063103593791Z_5c2a5e22

Populations: 160; settings that vary: none.
Main setting used below and in figures: {'rho': '0.5', 'phi': '0.5', 'K': '10'}.

## Main-setting results (means; see summary.csv for 95% CIs)

recommender  cat_entropy  homogeneity  welfare_net  fatigue  rounds  dropout
       bm25        0.302        0.203        0.046    1.288    20.0      0.0
    content        0.324        0.184        0.119    1.343    20.0      0.0
      ideal        0.537        0.093        2.204    1.232    20.0      0.0
         mf        0.722        0.365       -0.061    1.188    20.0      0.0
    mmr_0.6        0.451        0.093        0.527    1.299    20.0      0.0
       none        0.491        0.084        0.715    1.263    20.0      0.0
 popularity        0.547        0.683       -0.605    1.131    20.0      0.0
     random        0.579        0.043        0.082    1.181    20.0      0.0

## Files

- summary.csv
- paired_vs_none.csv
- paired_vs_random.csv
- figures/coverage_vs_utility.png
- figures/bubble_vs_echo_chamber.png

Paired differences compare each recommender with the reference on the same simulated
readers (matched by seed); a difference is marked significant when its bootstrap
95% CI excludes zero. Recommenders marked reference-only (ideal) use readers' true
preferences and are excluded from rank robustness.
