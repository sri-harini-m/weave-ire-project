# Final experiments

One command runs everything and writes report-ready tables and figures:

```bash
QUICK=1 bash scripts/run_final.sh     # 2 populations each: checks everything works (~5 min)
bash scripts/run_final.sh             # full runs (WORKERS=8 to use more cores)
ONLY="main types" bash scripts/run_final.sh   # rerun only some experiments
```

- Raw runs: `results/final/<experiment>/<timestamp>/` (git-ignored, keep on the server)
- Analysis: `reports/final/<experiment>/` (commit this; it is what the report uses)
- Configs: `configs/final/*.yaml`, generated from `config.yaml` by `scripts/make_final_configs.py`.
  Change settings in `config.yaml` or in `EXPERIMENTS` in that script, never by hand.

All runs use new users (recommenders see no reader history) unless stated, 10-article
slates, 100 real MIND readers per population and 20 rounds.

| Experiment | What varies | Populations | Question it answers | Main outputs |
|---|---|---|---|---|
| `main` | all recommenders, MMR lambda 0-1, fatigue phi 0/0.5/1 | 20 per cell | Which recommenders create filter bubbles, echo chambers and fatigue? How much diversity is needed? Does fatigue change the picture? | summary, paired_vs_none, mmr_diversity_curve, coverage_vs_utility, bubble_vs_echo_chamber |
| `types` | four reader types (cautious, explorer, easily bored, average), n0 = 20 | 20 | Who is hurt most by each recommender? | per_type.csv, reader_types.png |
| `slate` | slate size K = 5, 10, 25, 50, 500 | 20 | How much of "no recommender"'s advantage is just seeing more articles? | slate_size.png |
| `returning` | reader history visible to recommenders: yes / no | 20 | Does knowing history make bubbles better or worse? | paired_returning_vs_new.csv |
| `dropout` | readers may leave (threshold -1.5/-1/-0.5) x phi 0.5/1 | 20 | Which recommenders make readers quit? | summary (rounds, dropout) |
| `n0` | history-based uncertainty n0 = 5/20/80 | 20 | Sensitivity to n0 | summary, rank_robustness |
| `rho` | spillover rho = 0/0.5/0.9 | 20 | Sensitivity to learning spillover | summary, rank_robustness |
| `robustness` | gamma x beta x sigma_bar (27 settings) | 10 | Does the recommender ranking hold across reader assumptions? | rank_robustness.csv |
| `pca4`, `pca16` | article-similarity dimensions | 20 | Sensitivity to the embedding projection | summary |
| `validation` | none (real MIND data) | 10k impressions | Does the choice rule predict real clicks? Is there fatigue in real reading? | reader_choice_validation.*, fatigue_calibration.* |

Notes:
- With reader types on, each reader's own gamma/phi/decay replace the grid values;
  per-reader averages are reported as `reader_gamma`, `reader_phi`, `reader_fatigue_decay`.
- `ideal` uses readers' true preferences and is a reference point only (excluded from rank robustness).
- Dropout thresholds were chosen so a sizeable but not total share of readers leave; check
  `reports/final/dropout/summary.csv` and adjust in `make_final_configs.py` if almost
  nobody or almost everybody leaves.
- `analyse_results.py` works on any run: `python scripts/analyse_results.py <run folder> --out <dir>`.

## Analysis outputs (per experiment)

| File | Content |
|---|---|
| `summary.csv` | mean and 95% CI per setting x recommender |
| `paired_vs_none.csv`, `paired_vs_random.csv` | difference from the reference on the same readers (matched by seed), bootstrap 95% CI, significance |
| `paired_returning_vs_new.csv` | (returning) same recommender, history visible minus hidden |
| `per_type.csv` | (types) every metric per reader type |
| `rank_robustness.csv` | Kendall tau of recommender rankings between settings, and which recommender is best how often |
| `figures/` | coverage vs utility, filter bubble vs echo chamber, MMR curve, slate size, reader types |
| `README.md` | headline table for the main setting and a list of files |

## Where each result goes in the report

| Report section | Source |
|---|---|
| Simulator validation | `reports/figures/repro_*` (paper reproduction), `reports/world_coherence.md`, `reports/final/validation/reader_choice_validation.md` |
| Fatigue in real reading | `reports/final/validation/fatigue_calibration.md` |
| Main comparison | `reports/final/main/` |
| Diversity (MMR) | `reports/final/main/figures/mmr_diversity_curve.png` |
| Reader types | `reports/final/types/` |
| Choice-set size | `reports/final/slate/` |
| New vs returning users | `reports/final/returning/paired_returning_vs_new.csv` |
| Session dropout | `reports/final/dropout/` |
| Robustness / limitations | `reports/final/{robustness,rho,n0,pca4,pca16}/` |
