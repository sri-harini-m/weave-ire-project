# Saved experiment comparison

Rebuilt from saved arrays. No simulator was rerun.

Condition IDs map to every saved parameter in conditions.csv.
Rows with n_populations=1 have no estimated confidence interval.
Synthetic rows are engineering checks, not MIND research findings.
Exposure metrics describe explicit returned slates. Full-catalogue browsing has no measured slate exposure.

| condition_id | recommender | rho | phi | diversity_weight | data_kind | n_populations | cat_entropy | diversity | homogeneity | fatigue | welfare_net | catalog_coverage | slate_ild |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C000 | none | 0.5 | 0.0 | NA | synthetic | 3 | 0.6839 ± 0.0479 | 0.6926 ± 0.0136 | 0.08965 ± 0.0102 | 0 ± 0 | 1.376 ± 0.92 | 0.8111 ± 0.0288 | NA |
| C001 | random | 0.5 | 0.0 | NA | synthetic | 3 | 0.7549 ± 0.0257 | 0.7331 ± 0.00552 | 0.07789 ± 0.0108 | 0 ± 0 | 0.906 ± 0.934 | 0.8167 ± 0.0327 | 0.7677 ± 0.00264 |
| C002 | popularity | 0.5 | 0.0 | NA | synthetic | 3 | 0.7641 ± 0.0355 | 0.7716 ± 0.00575 | 0.5215 ± 0.0187 | 0 ± 0 | 0.01058 ± 1.11 | 0.2 ± 3.85e-17 | 0.7811 ± 0.0069 |
| C003 | content | 0.5 | 0.0 | NA | synthetic | 3 | 0.6625 ± 0.0186 | 0.6697 ± 0.00988 | 0.08766 ± 0.0132 | 0 ± 0 | 1.308 ± 1.01 | 0.8111 ± 0.0109 | 0.706 ± 0.0103 |
| C004 | bm25 | 0.5 | 0.0 | NA | synthetic | 3 | 0.1715 ± 0.0125 | 0.7295 ± 0.00304 | 0.1254 ± 0.0364 | 0 ± 0 | 0.5973 ± 0.848 | 0.7333 ± 0.0943 | 0.7197 ± 0.00565 |
| C005 | mf | 0.5 | 0.0 | NA | synthetic | 3 | 0.8169 ± 0.0519 | 0.7394 ± 0.0088 | 0.2968 ± 0.0216 | 0 ± 0 | 0.6692 ± 0.929 | 0.3333 ± 0 | 0.7641 ± 0.00581 |
| C006 | mmr_0 | 0.5 | 0.0 | 0.0 | synthetic | 3 | 0.6625 ± 0.0186 | 0.6697 ± 0.00988 | 0.08766 ± 0.0132 | 0 ± 0 | 1.308 ± 1.01 | 0.8111 ± 0.0109 | 0.706 ± 0.0103 |
| C007 | mmr_0.3 | 0.5 | 0.0 | 0.3 | synthetic | 3 | 0.6536 ± 0.0451 | 0.677 ± 0.0132 | 0.0912 ± 0.00718 | 0 ± 0 | 1.361 ± 0.946 | 0.8056 ± 0.0714 | 0.7345 ± 0.0122 |
| C008 | mmr_0.6 | 0.5 | 0.0 | 0.6 | synthetic | 3 | 0.6713 ± 0.0438 | 0.6832 ± 0.00834 | 0.09157 ± 0.00804 | 0 ± 0 | 1.38 ± 0.959 | 0.8167 ± 0.0327 | 0.7613 ± 0.00922 |
| C009 | none | 0.5 | 0.5 | NA | synthetic | 3 | 0.7393 ± 0.0546 | 0.7076 ± 0.00364 | 0.07488 ± 0.00628 | 0.8827 ± 0.0023 | -0.03015 ± 1.16 | 0.8444 ± 0.0218 | NA |
| C010 | random | 0.5 | 0.5 | NA | synthetic | 3 | 0.7558 ± 0.0168 | 0.7418 ± 0.00328 | 0.07737 ± 0.0125 | 0.8697 ± 0.00124 | -0.5847 ± 1.05 | 0.8222 ± 0.0218 | 0.7684 ± 0.00239 |
| C011 | popularity | 0.5 | 0.5 | NA | synthetic | 3 | 0.7783 ± 0.0285 | 0.7746 ± 0.0047 | 0.5076 ± 0.00867 | 0.8565 ± 0.00178 | -1.052 ± 1.04 | 0.2 ± 3.85e-17 | 0.7824 ± 0.0045 |
| C012 | content | 0.5 | 0.5 | NA | synthetic | 3 | 0.6975 ± 0.041 | 0.6737 ± 0.0021 | 0.0745 ± 0.00952 | 0.8973 ± 0.00178 | -0.1841 ± 1.03 | 0.8389 ± 0.0288 | 0.707 ± 0.0073 |
| C013 | bm25 | 0.5 | 0.5 | NA | synthetic | 3 | 0.1689 ± 0.0488 | 0.7285 ± 0.00272 | 0.08739 ± 0.0111 | 0.8751 ± 0.00126 | -0.8433 ± 1.26 | 0.8444 ± 0.0218 | 0.7151 ± 0.00607 |
| C014 | mf | 0.5 | 0.5 | NA | synthetic | 3 | 0.8418 ± 0.0391 | 0.7501 ± 0.00441 | 0.2885 ± 0.00605 | 0.8668 ± 0.00244 | -0.9198 ± 1.11 | 0.3389 ± 0.0218 | 0.7709 ± 0.00669 |
| C015 | mmr_0 | 0.5 | 0.5 | 0.0 | synthetic | 3 | 0.6975 ± 0.041 | 0.6737 ± 0.0021 | 0.0745 ± 0.00952 | 0.8973 ± 0.00178 | -0.1841 ± 1.03 | 0.8389 ± 0.0288 | 0.707 ± 0.0073 |
| C016 | mmr_0.3 | 0.5 | 0.5 | 0.3 | synthetic | 3 | 0.6944 ± 0.00783 | 0.6832 ± 0.00409 | 0.07904 ± 0.00432 | 0.8937 ± 0.00183 | -0.1588 ± 0.986 | 0.8 ± 0.0327 | 0.7369 ± 0.00309 |
| C017 | mmr_0.6 | 0.5 | 0.5 | 0.6 | synthetic | 3 | 0.7136 ± 0.025 | 0.6939 ± 0.00241 | 0.07467 ± 0.0102 | 0.8895 ± 0.00106 | -0.1065 ± 1.11 | 0.8278 ± 0.0109 | 0.7652 ± 0.00331 |

Full numeric results, effective sample counts, and CI widths are in summary.csv.
Plots use summary.csv and curves_summary.csv. Regenerate these outputs from the same archives when presentation needs change.
