# Saved artifacts for report preparation

Every new `run_news.py` or `run_report.py` run creates a unique child folder under
`--out`. It never reuses an older run folder. Files remain local under `results/`,
which Git ignores. Back up selected finished run folders before changing machines.

The current algorithms are classical NumPy recommenders. Their saved models
contain settings, learned popularity counts, catalogue fingerprints, and exact
random-generator states. Content embeddings are saved once in each population's
world archive. MMR similarity is rebuilt from those saved embeddings on reload.
There are no LLM calls or LLM responses in this implementation.

## What each run saves

```text
results/<output-base>/<unique-run>/
  run.json                     arguments, config, data kind, status, timestamps
  run.log                      command stdout, stderr, and failure tracebacks
  environment.json             Python, platform, installed package versions
  config.json                  effective input configuration
  git_revision.txt             repository commit
  git_status.txt               branch and working-tree status
  working_tree.patch           tracked changes relative to HEAD
  source/                      source, scripts, tests, docs and dependency file
  input_files.json             original processed-data paths, sizes and hashes
  manifest.json                saved-file SHA256 hashes and sizes
  expected_populations.json     planned jobs, conditions, population IDs and seeds
  original_populations.csv     original runner population results
  original_curves.csv          original runner per-round curves
  populations/job_000000/
    metadata.json              condition, population, seed, settings, reader IDs
    params.json                complete simulator parameters
    status.json                created/running/complete/failed, progress, error
    population.log             per-population progress
    world.npz                  article IDs/categories/embeddings/distances/covariance
    world.json                 world type and fingerprint
    article_metadata.json      selected article records, when available
    initial_observable.npz     initial reader profiles and seen-item masks
    initial_hidden.npz         diagnostic true values and starting beliefs
    models/initial.npz
    models/before_0000.npz      model immediately before recommendation
    models/after_0000.npz       model after feedback
    models/final.npz
    rounds/round_0000_input.npz availability, activity, ratings, profiles, diagnostics
    rounds/round_0000_input.json history and additional visible context
    rounds/round_0000_output.npz slates, chosen items, feedback and diagnostics
    responses.jsonl            recommendation/reader response trace, flushed per round
    run.npz                    final choices, true values, penalties, exit rounds
    user_metrics.csv
    curves.csv
  report/
    conditions.csv             every condition parameter, mapped to a condition ID
    populations.csv            one metric row per population
    users.csv                  one row per reader and population
    curves.csv                 one row per round and population
    summary.csv                population means, effective counts and 95% CI widths
    curves_summary.csv         mean curves and population-level CI widths
    summary.md                 report comparison table
    metrics_definitions.json   formulas, denominators and interpretation limits
    figures/*.png              raster figures
    figures/*.svg              editable vector figures
```

Some root files apply only to one runner. `run_news.py` additionally retains its
original summary and figure. A failed run may contain only completed and partial
population files. The report builder refuses incomplete population collections.
It also refuses failed run bundles and mismatched saved input hashes.

Hidden diagnostic arrays remain separate from observable recommender inputs.
Saving them does not expose them to a recommendation algorithm.

All NumPy loads use `allow_pickle=False`. Checkpoints only restore registered
algorithms. They reject a different catalogue or changed article ordering.

## Metrics available for the report

| Area | Metrics |
| --- | --- |
| Reading breadth | Category entropy, category count, dominant-category share, consecutive-read distance, pairwise reading diversity |
| Reader overlap | Mean pairwise Jaccard article overlap (`homogeneity`) |
| Fatigue | Mean fatigue penalty and per-round fatigue curves |
| Utility | Mean gross/net welfare, cumulative gross/net welfare, total net utility |
| Participation | Explicit dropout share, exit round, actual read count, consumption share, active share |
| Catalogue use | Consumed-article coverage, consumption Gini and HHI concentration |
| Offered slates | Intra-list diversity, category coverage and entropy |
| Exposure | Explicit-slate catalogue coverage, exposure count, Gini and HHI concentration |

Consumption and exposure are different measurements. A diverse slate does not
guarantee diverse reading. The `none` baseline has no recorded rendered slate;
its exposure metrics are unavailable. We do not invent exposure events from the
full browsing catalogue.

Consumption share is reads divided by possible reader-rounds. It is not CTR.
Overlap measures shared articles. It does not establish ideological agreement.
Exhausting articles does not count as explicit dropout. Default readers do not
quit from fatigue unless an `outside_option` threshold is configured.

Each condition keeps all simulator parameters and recommender settings separate.
Confidence intervals use population means, not individual reads as independent
samples. Current intervals use `1.96 * sample_std / sqrt(n)`. With one population,
the interval is unavailable. This approximation can be weak for small samples.

## Run the synthetic archive check

Activate Python 3.10 or newer and install `requirements.txt`. The local checked
environment uses Python 3.12. These commands run from the repository root:

```bash
python -m pytest -q
python scripts/run_report.py --out results/report_checks
```

The default check uses 12 readers, 60 generated articles, eight rounds, three
populations, two fatigue levels, and three MMR weights. It compares seven
algorithm/settings combinations. These are synthetic engineering checks,
not MIND findings for the project report. Figures identify synthetic inputs.

The script also supports `--users 100 --articles 500 --rounds 20`. Saving every
round uses more disk space and time than the previous timing-only benchmark.
Save only the actual experiments you need, but keep their complete run folders.

## Run the real comparison when Person A supplies data

This checkout currently lacks `src/data/store.py` and the processed MIND files.
The real-data command cannot complete until they are supplied.

```bash
python scripts/run_news.py \
  --recommenders none random popularity content mmr \
  --mmr-diversity-weights 0 0.3 0.6 \
  --n-pop 20 --workers 2 --out results/news
```

The script prints the unique saved directory. It saves selected article metadata
and sampled reader IDs/history when the data store provides them. It fingerprints
the original processed files and stores the exact arrays used by the simulation.
It does not copy the entire MIND download or download embedding-model weights.
Those are not needed to rebuild figures from an already saved population.

## Rebuild tables and figures without another simulation

```bash
python scripts/run_report.py --rebuild results/news/<unique-run>
```

This reads saved population arrays and round responses. It regenerates only the
`report/` outputs and updates the file manifest. It does not call a recommender,
draw new readers, or overwrite raw population archives.
Before rebuilding, it verifies existing raw, source, config and log fingerprints.
Changed report files are allowed because they are derived presentation outputs.

For a new metric, extend `population_report()` in `src/sim/reporting.py`, then
run the rebuild command. Saved world data, slates, choices, utilities and penalties
support many later metrics. A metric needing information that was never modeled
cannot be recovered; ideological labels and real click probabilities are examples.

## Reload a saved recommendation model

```python
from pathlib import Path
from src.sim.artifacts import load_checkpoint, load_state, load_world

population = Path("results/news/<unique-run>/populations/job_000000")
world = load_world(population)
state = load_state(population, t=0, world=world)
model = load_checkpoint(population / "models/before_0000.npz", world)
slate = model.recommend(state)
```

This reproduces the saved recommendation, including random-baseline RNG state.
Loading a final model also needs a compatible new observable state for prediction.
This feature replays recommendations. It does not resume the complete simulator
after interruption. Partial archives remain available for diagnosis.

## Report-preparation checklist

1. Keep the complete finished run directory and its manifest.
2. Verify `run.json` and all population statuses say `complete`.
3. Use `report/summary.md` for the comparison and CSVs for exact numbers.
4. Use PNGs directly or edit SVGs for the report layout.
5. Cite the seed, config, data kind, source revision and number of populations.
6. Keep synthetic engineering results separate from real-data findings.
7. State the metric limitations and simulator assumptions alongside conclusions.
