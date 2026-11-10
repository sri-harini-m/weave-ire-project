# Recommendation study: controls, feedback and saved comparisons

## Repository inventory checked on November 10, 2026

Remote branches were `main`, `sbidisha_dev`, and `akshara`. Main already merged
the simulator/data pipeline and existing recommenders.
The development branch was fast-forwarded to main commit `c59e0d9` before this work.
Akshara's branch contains the earlier Person A implementation.

Existing components include the MIND preparation/store modules, reader validation,
circle reproduction checks, news metrics, history visibility flag, and published
Person A reports. The processed MIND arrays are Git-ignored and are not in any
remote branch. Existing published results must not be presented as reruns of the
new controls in this document.

## Requested controls

| Control | Implementation |
| --- | --- |
| Slate size | `--k 10` or `--slate-sizes 5 10 25 50 500` |
| Extended MMR sweep | `--mmr-diversity-weights 0 0.3 0.6 0.9 1.0` for news; `--mmr-weights` for synthetic runs |
| Initial history visibility | `--reader-history-visible false`, `true`, or `both` |
| Model snapshots | `--snapshot-policy final` (default), `all`, or `none` |
| Full diagnostic arrays | `--archive-diagnostics`; omitted by default |
| Reader departures | Optional `--outside-option VALUE`; default reader behavior is retained |

The `reader_history_visible` flag already existed in main. The new controls make
it convenient to compare both settings with the same populations and catalogue.
Initial reader beliefs remain grounded in the same profiles. Only the initial
information available to recommenders changes. Simulated histories still start
empty, and subsequent simulated reads remain visible in both settings.

Returning-mode inputs can include embedding profiles, historical article text,
and category counts. Initial category counts use the order `np.unique(world.cats)`.
The news adapter builds them from each sampled reader's known MIND history.
New-user recommenders receive none of these initial history fields.

The seed excludes recommender name, slate size and history visibility. Thus these
conditions share sampled users and true values. Fatigue strengths keep the
existing separate-seed convention; comparisons between recommenders at the same
fatigue setting remain matched. Saved population seeds make this explicit.

MMR retains the existing formula `relevance - lambda * maximum_slate_similarity`.
Here lambda is a diversity penalty: zero equals content ranking, and one is the
largest supported penalty. One is not a pure-diversity objective under this formula.

## Privileged ideal reference

`ideal` ranks eligible articles by each reader's true gross utility. It follows
the role of the ideal reference in [Chaney et al.](https://arxiv.org/abs/1710.11214).
It does not reproduce that paper's complete interaction model.

Only the exact `IdealRecommender` class receives true values through
`set_reference_values()`. They never enter ordinary `SimState.extra` or observable
feedback. Reports and model headers mark this method `reference_only=true`.

Readers still choose within the slate. Therefore ideal need not upper-bound
realized net welfare with K greater than one. Fatigue and reader beliefs still
matter. With K=1, the initial selected article maximizes available gross utility.
With K at least N, all methods offer the whole eligible catalogue; the current
reader model ignores slate ordering, so choices should match full browsing.

## Observable fatigue feedback

After a consumed article, the simulator emits a coarse boredom flag:

```text
boredom = 1 if experienced fatigue penalty > boredom_threshold else 0
```

The default threshold is 0.25 utility units. This is a modeled self-report,
not a boredom label measured in MIND. The threshold is configurable and saved.
It deliberately does not expose exact latent fatigue, unobserved utility or
counterfactual values for unread articles.

Opted-in methods receive:

```python
observe(state, users, items, net_feedback,
        signals={"boredom": flags_for_consumed_items,
                 "left_users": readers_who_left_this_round})
```

Ordinary methods keep the existing four-argument callback. A reader departure is
recorded even when nobody reads. It does not become a dislike for every unconsumed
article in the slate. Exhaustion is also not automatically treated as boredom.

| Method | Use of observable feedback |
| --- | --- |
| `fatigue_popularity` | Each read adds `1 - fatigue_weight * boredom` to popularity |
| `fatigue_mf` | Fits `net_feedback - fatigue_weight * boredom`; popularity fallback also downweights bored reads |
| `fatigue_content` | Penalizes categories recently consumed; bored reads contribute extra recent mass |

MF's target adjustment is an additional preference for avoiding reported boredom.
Its net feedback already includes fatigue. The adjusted target does not estimate
pre-fatigue utility. Fatigue-blind comparisons mean *no explicit boredom signal*;
their readers and observed net feedback still experience fatigue.

Fatigue content decays category mass over rounds. A consumed category receives
`1 + boredom`; the penalty is recent category mass divided by `1 + total mass`,
scaled by `fatigue_weight`. Empty rounds still advance decay. Only observable
categories, reads and coarse reports enter this model.

Weights of zero reproduce the corresponding blind rankings and feedback targets.
`fatigue_mf` inherits ordinary MF settings unless specifically overridden, so a
comparison does not silently change factor counts or training iterations.

The feedback design is motivated by [Cao et al.](https://arxiv.org/abs/2008.09733).
It does not implement their dependent-click model or bandit guarantees.

## Calibrated recommendation

`calibrated` offers a second diversity method. Its target distribution combines
visible historical category counts with simulated reads. With no known category
history it uses the public catalogue mix. It never uses hidden true preferences.

The greedy selection score is:

```text
(1 - calibration_weight) * content_relevance
    - calibration_weight * KL(target || 0.99 * proposed_slate_mix + 0.01 * target)
```

This follows the distribution-matching approach of
[Steck's calibrated recommendations](https://doi.org/10.1145/3240323.3240372).
Calibration matches a reader's known topic mix. It does not necessarily maximize
topic entropy or require equal representation of every category.
`calibration_weight=0` exactly matches ordinary content ranking.

## Archive size and recovery trade-off

The default retains only the final model, observable round inputs, response
traces, raw metric arrays, settings and reports. Hidden belief matrices, latent
utility arrays and exact fatigue states are omitted unless diagnostics are enabled.
The ideal final model is intentionally privileged and explicitly labeled.

Identical world arrays are saved once per rho. Population paths hardlink the
immutable shared file, so offline loaders remain compatible without allocating
the same distance/covariance matrices repeatedly. Cross-device copying is a fallback.
Copying a run folder with software that does not preserve hardlinks may use more
disk space; a tar archive normally preserves them.

Use `--snapshot-policy all --archive-diagnostics` for complete debugging/replay.
With final-only snapshots, past inputs/responses remain inspectable but pre-round
model replay is unavailable. Final model reload and offline report rebuild work.

## Real-data experiment command

After preparing MIND with the frozen MiniLM embeddings:

```bash
python scripts/run_news.py --config configs/recommendation_study.yaml \
  --recommenders none random popularity content bm25 mf mmr ideal calibrated \
                 fatigue_popularity fatigue_mf fatigue_content \
  --slate-sizes 5 10 25 50 500 \
  --mmr-diversity-weights 0 0.3 0.6 0.9 1.0 \
  --reader-history-visible both --snapshot-policy final \
  --workers 2 --out results/recommendation_study
```

The full config uses 100 readers, 20 rounds and 20 populations. The pilot config
uses eight readers, six rounds and two populations. Both retain a 500-article
catalogue so the K=500 condition actually exposes that catalogue. Pilot results
are preliminary; they do not replace the full study or establish an optimal lambda.

To run the pilot, replace the config with `configs/recommendation_pilot.yaml`.
For synthetic engineering checks, use `scripts/run_report.py` with `--articles 500`
and equivalent controls. Synthetic outputs remain explicitly labeled.

## Files for report preparation

The completed November 10 pilot has 640 population runs across 320 conditions.
Its [report pack](../reports/recommendation_pilot/README.md) includes tables,
settings, PNG/SVG figures, data provenance and verification evidence.
The pilot uses real MIND inputs with simulated reader outcomes, not measured human fatigue.
The full 20-population study remains a separate run.

The run folder saves `settings_used.json`, exact per-condition settings,
population/reader metrics, curves, logs, source and dependency versions.
`report/summary.csv` and `summary.md` contain confidence intervals and effective
population counts. Dedicated comparison tables cover K, MMR, history visibility,
fatigue feedback and diversity methods. Paired tables compare matched populations.

Rebuild without another simulation:

```bash
python scripts/run_report.py --rebuild <saved-run-directory>
```

Integrity checks run before rebuilding. Failed or missing planned populations
block a final comparison. Report plots keep simulation and model settings
separate; large condition grids use compact study views.
