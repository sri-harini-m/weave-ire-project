# Analysis of 20261112T161012510966Z_25bc50ca

Populations: 240; settings that vary: none.
Main setting used below and in figures: {'rho': '0.5', 'K': '10'}.

## Main-setting results (means; see summary.csv for 95% CIs)

       recommender  cat_entropy  homogeneity  welfare_net  fatigue  rounds  dropout
              bm25        0.307        0.212        0.149    1.156    20.0      0.0
        calibrated        0.206        0.217        0.168    1.245    20.0      0.0
           content        0.326        0.189        0.200    1.256    20.0      0.0
   fatigue_content        0.750        0.145        0.587    1.085    20.0      0.0
        fatigue_mf        0.726        0.371        0.218    0.998    20.0      0.0
fatigue_popularity        0.545        0.678       -0.365    0.867    20.0      0.0
             ideal        0.569        0.092        2.453    1.087    20.0      0.0
                mf        0.730        0.383        0.224    1.002    20.0      0.0
           mmr_0.6        0.465        0.096        0.681    1.155    20.0      0.0
              none        0.525        0.091        0.906    1.059    20.0      0.0
        popularity        0.545        0.678       -0.365    0.867    20.0      0.0
            random        0.593        0.043        0.295    0.964    20.0      0.0

## Files

- summary.csv
- paired_vs_none.csv
- paired_vs_random.csv
- per_type.csv
- figures/coverage_vs_utility.png
- figures/bubble_vs_echo_chamber.png
- figures/reader_types.png

Paired differences compare each recommender with the reference on the same simulated
readers (matched by seed); a difference is marked significant when its bootstrap
95% CI excludes zero. Recommenders marked reference-only (ideal) use readers' true
preferences and are excluded from rank robustness.
