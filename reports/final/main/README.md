# Analysis of 20261112T144749283363Z_dda369ca

Populations: 960; settings that vary: phi.
Main setting used below and in figures: {'rho': '0.5', 'phi': '0.5', 'K': '10'}.

## Main-setting results (means; see summary.csv for 95% CIs)

       recommender  cat_entropy  homogeneity  welfare_net  fatigue  rounds  dropout
              bm25        0.310        0.200        0.016    1.105    20.0      0.0
        calibrated        0.209        0.211        0.048    1.191    20.0      0.0
           content        0.320        0.185        0.064    1.203    20.0      0.0
   fatigue_content        0.746        0.139        0.419    1.040    20.0      0.0
        fatigue_mf        0.718        0.361        0.046    0.944    20.0      0.0
fatigue_popularity        0.540        0.688       -0.512    0.827    20.0      0.0
             ideal        0.534        0.094        2.371    1.035    20.0      0.0
                mf        0.718        0.368        0.046    0.951    20.0      0.0
             mmr_0        0.320        0.185        0.064    1.203    20.0      0.0
           mmr_0.3        0.400        0.110        0.380    1.159    20.0      0.0
           mmr_0.6        0.453        0.091        0.551    1.119    20.0      0.0
           mmr_0.9        0.488        0.087        0.597    1.091    20.0      0.0
             mmr_1        0.497        0.088        0.591    1.084    20.0      0.0
              none        0.509        0.089        0.786    1.040    20.0      0.0
        popularity        0.540        0.688       -0.512    0.827    20.0      0.0
            random        0.584        0.043        0.157    0.928    20.0      0.0

## Files

- summary.csv
- paired_vs_none.csv
- paired_vs_random.csv
- rank_robustness.csv
- figures/coverage_vs_utility.png
- figures/bubble_vs_echo_chamber.png
- figures/mmr_diversity_curve.png

Paired differences compare each recommender with the reference on the same simulated
readers (matched by seed); a difference is marked significant when its bootstrap
95% CI excludes zero. Recommenders marked reference-only (ideal) use readers' true
preferences and are excluded from rank robustness.
