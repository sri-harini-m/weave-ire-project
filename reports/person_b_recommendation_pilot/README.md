# Person B: recommendation pilot

This pilot uses real MIND-small articles and histories with simulated readers.
Utilities, fatigue, boredom reports, and departures come from the simulator.
They are not measured human outcomes from MIND.

## Completed experiment

- 500 articles, 14 categories, eight readers, and six rounds per population.
- Two populations per condition; 320 conditions and 640 saved population runs.
- Requested K: 5, 10, 25, 50, and 500.
- Initial history: hidden and visible, using matched populations.
- Fatigue strength: 0 and 0.5; correlation rho: 0.5.
- MMR diversity penalty: 0, 0.3, 0.6, 0.9, and 1.
- Twelve methods, including five MMR settings, produce sixteen configurations.
- The departure threshold is zero. The boredom threshold is 0.25.
- Final model snapshots and observable responses remain saved locally.

The methods are `none`, `random`, `popularity`, `content`, `bm25`, `mf`, `mmr`,
`ideal`, `calibrated`, `fatigue_popularity`, `fatigue_mf`, and `fatigue_content`.

The ideal reference ranks true gross utility. It has privileged information.
It does not guarantee maximum realized net welfare when readers choose within a slate.
Ordinary methods receive observable feedback only.

## Data and validation

The preparation pipeline merged 65,238 articles from MIND-small train and dev news.
It extracted 32,974 eligible histories from train behaviors.
MiniLM produced normalized 384-dimensional embeddings. No fake embeddings were used.
`data_provenance.json` records the model revision and processed-file hashes.

The complete test suite passed: **233 tests, zero warnings**.
`tests.log` contains the output.
Archive validation and offline rebuild evidence appear in `verification/`.

The comparison coverage is complete:

| Contrast | Matched population pairs | Unmatched |
| --- | ---: | ---: |
| Returning minus new users | 320 | 0 |
| Fatigue-aware minus blind | 120 | 0 |
| Method minus full browsing | 600 | 0 |

## Saved report files

`summary.csv` preserves exact conditions, metric means, population counts, and confidence intervals.
`settings_used.json`, `config.json`, and `conditions.csv` record the settings.
`environment.json` records installed package versions.
The metric definitions describe each calculation and its limits.

Use `slate_size_comparison.csv`, `mmr_sweep.csv`, and the history/fatigue comparison tables.
The `paired_*` tables contain differences between matched population runs.
Each difference equals the left condition minus the right condition.
The paired summaries retain all other simulator and model settings.

`figures/` includes PNG and SVG plots.
Their companion CSV files identify exact plotted conditions, lines, and facets.
The no-recommender baseline browses the full catalogue at every requested K.
It does not measure rendered slate exposure.

This small pilot verifies the experiment pipeline.
Two populations provide limited precision and do not establish the best MMR penalty.
The full study configuration uses 100 readers, 20 rounds, and 20 populations.
That full study has not been run here.

## Reproduce or rebuild

Run from the repository root after preparing MIND:

```bash
python scripts/run_news.py --config configs/recommendation_pilot.yaml \
  --recommenders none random popularity content bm25 mf mmr ideal calibrated \
                 fatigue_popularity fatigue_mf fatigue_content \
  --slate-sizes 5 10 25 50 500 \
  --mmr-diversity-weights 0 0.3 0.6 0.9 1 \
  --reader-history-visible both --snapshot-policy final \
  --outside-option 0 --workers 2 --out results/recommendation_pilot
```

The complete local archive is:
`results/recommendation_pilot/20261110T151206823348Z_3bd26945`.
It contains final models, observable round inputs, response traces, metric arrays,
source snapshots, logs, and file manifests. Data and complete archives are Git-ignored.

Rebuild its reports without simulating readers again:

```bash
python scripts/run_report.py --rebuild \
  results/recommendation_pilot/20261110T151206823348Z_3bd26945
```

Use `configs/recommendation_study.yaml` for the larger study.
See [implementation notes](../../docs/recommendation_study.md) for algorithm details and references.
