# Analysis of 20261113T060414053122Z_496b220d

Populations: 160; settings that vary: none.
Main setting used below and in figures: {'rho': '0.5', 'phi': '0.5', 'K': '10'}.

## Main-setting results (means; see summary.csv for 95% CIs)

recommender  cat_entropy  homogeneity  welfare_net  fatigue  rounds  dropout
       bm25        0.315        0.202        0.485    0.722    20.0      0.0
    content        0.328        0.184        0.553    0.868    20.0      0.0
      ideal        0.567        0.096        2.940    0.653    20.0      0.0
         mf        0.729        0.383        0.508    0.582    20.0      0.0
    mmr_0.6        0.468        0.095        0.997    0.720    20.0      0.0
       none        0.540        0.090        1.196    0.591    20.0      0.0
 popularity        0.539        0.696       -0.040    0.400    20.0      0.0
     random        0.592        0.044        0.649    0.483    20.0      0.0

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
