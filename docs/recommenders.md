# Person B: recommender inputs and implementation

See [report_artifacts.md](report_artifacts.md) for saved models, responses, logs,
metrics, figures, and rebuilding reports without another simulation.

This implementation uses the existing simulator's NumPy interface. It requires
Python 3.10 or later. No external recommender service or model download is needed.
The adapter from Person A's final data format can change without rewriting the
ranking algorithms.

## Current algorithms

| Registry name | Ranking | Updates |
| --- | --- | --- |
| `none` | Reader sees the entire catalogue | None |
| `random` | Random eligible articles | Seeded generator |
| `popularity` | Simulated read counts plus an optional initial prior | Every observed read adds one count |
| `content` | Cosine similarity to a visible profile and simulated reading history | Rebuilds the profile from the current history |
| `mmr` | Greedy content relevance minus a slate redundancy penalty | Uses the same history and popularity updates as `content` |
| `bm25` | Keyword relevance to visible query text and simulated reading history | Reuses static article text; updates popularity fallback |
| `mf` | Predicted observed utility from user/item factors and biases | Refits on accumulated observed feedback after each round |

These algorithms do not receive unobserved utilities or Bayesian beliefs.
The `ratings` field contains observed feedback only. Popularity, content and MMR
count reads without weighting content profiles by feedback. Matrix factorisation
fits the observed utility values, including negative and zero values.

## Array contract

`U` is the reader count, `N` the article count, and `D` the embedding dimension.
User row `u` and article index `n` must keep the same meaning throughout one run.
External article IDs map through `world.ids[n]`; slates contain indices, not IDs.

| Location | Shape / type | Meaning |
| --- | --- | --- |
| `world.profile_vecs` | float `[N, D]` | Article embeddings; required by `content` and `mmr` |
| `world.popularity` | nonnegative float `[N]`, or `None` | Optional MIND popularity prior |
| `state.available` | bool `[U, N]` | Authoritative unread/eligible mask; exclude pre-session and simulated reads |
| `state.history` | list of `U` lists of integer indices | Simulated reads, oldest first; empty initially |
| `state.ratings` | float `[U, N]` | Observed utility; `NaN` for unobserved pairs |
| `state.active` | bool `[U]` | Whether each reader remains active |
| `state.extra['profiles']` | finite float `[U, D]`, optional | Visible initial profiles, in the article embedding space |
| `world.article_metadata` | `N` records with `news_id`, `title`, optional `abstract` | Required BM25 article text; aligned through catalogue IDs |
| `state.extra['query_texts']` | optional list of `U` strings | Observable initial keyword context for BM25 |
| `state.t` | integer | Current round |
| Returned slate | integer `[U, K]` | Distinct eligible article indices; `-1` pads empty positions |

Do not put hidden preferences, future clicks, or unobserved utilities in `extra`.
An initial profile is an observable input, not the simulator's hidden true tastes.
If Person A uses interests without historical clicks, pass those visible interests
as profiles only when both sides agree they are available to the recommender.

Missing profiles are supported. Invalid profile dimensions raise `ValueError`.
Zero embedding rows remain zero. Nonfinite embeddings raise `ValueError`.
Readers with no usable profile or history use the current popularity ranking.
Missing article embeddings raise `ValueError` for content and MMR; there is no
silent substitution of the simulator's latent correlation matrix.

The simulator now forwards its optional `profiles` argument into `extra`.
It also forwards optional `query_texts`. The news-user adapter builds these texts
from each sampled user's visible MIND history titles and abstracts. It does not
turn embedding coordinates or hidden preferences into keywords.
Existing calls without either input continue to work.

## Lifecycle

1. Call `reset(world, U, rng)` once before each population.
2. Call `recommend(state)` once per round, before readers consume anything.
3. Update availability, history, ratings, and active status in the simulator.
4. Call `observe(state, users, items, feedback)` once for that round's reads.

The last three arrays are aligned one-dimensional arrays. Their length is the
number of observed reads. Multiple readers may read the same article. Each read
adds one popularity count, including reads with zero or negative utility.
`reset` removes learned counts from the previous population.
`recommend` does not update learned counts. Repeated recommendation calls with
unchanged inputs are deterministic for popularity, content, and MMR.

All slate recommenders return `-1` for inactive readers. Ties use the lower
article index. Exhausted readers and `K > N` receive padding. `K = 0` returns
`[U, 0]`. The `none` baseline returns `None`, preserving full-catalogue browsing.

## Defaults and equations

Popularity starts from:

```text
prior = world.popularity / sum(world.popularity)
counts = prior_weight * prior + observed_simulated_read_counts
```

An absent or all-zero prior contributes zero counts. The default prior has total
mass one. MIND popularity may already be log-scaled; it is not a raw click count.

Content ranking uses normalized embeddings and a normalized centroid:

```text
centroid[u] = profile_weight * unit(initial_profile[u])
              + sum(unit(article_embedding[n]) for n in history[u])
relevance[u, n] = unit(centroid[u]) dot unit(article_embedding[n])
```

The default initial profile has the same weight as one simulated read.
Set `profile_weight = 0` to ignore initial profiles. No profile uses a zero vector.
Cold-start relevance is popularity divided by its maximum, or zero without reads
or a prior. This scaling preserves popularity ordering and bounds those scores.

MMR first selects the most relevant eligible article. Each later selection uses:

```text
score[n] = relevance[n] - diversity_weight * max(cosine(n, j) for j in selected)
```

`diversity_weight` ranges from zero to one. Zero exactly matches content ranking.
Larger values penalize similarity more. Negative cosine similarities remain
negative, so dissimilar articles can receive a novelty bonus.
This parameter is the brief's diversity-penalty convention. Some MMR sources use
lambda as a relevance weight, where one disables diversity. Do not compare these
parameters numerically without converting the equations.

Cosine scoring follows the [scikit-learn definition](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html).
The greedy redundancy approach follows the [Personalized Machine Learning book's MMR example](https://cseweb.ucsd.edu/~jmcauley/pml/code/chap10.html),
with the explicitly different penalty convention above. This code is a NumPy
implementation; it does not add either source's libraries as dependencies.

## Checks and integration

```bash
python -m pytest -q
python scripts/check_recommenders.py
python scripts/run_news.py --help
```

The synthetic benchmark is an engineering check, not evidence about MIND users,
filter bubbles, echo chambers, or welfare. Do not report synthetic outputs as
project findings.

Once Person A supplies `src/data/store.py` and processed MIND data:

```bash
python scripts/run_news.py --recommenders none random popularity content mmr bm25 mf --n-pop 20
python scripts/run_news.py --recommenders mmr --mmr-diversity-weight 0 --out results/mmr_0
python scripts/run_news.py --recommenders mmr --mmr-diversity-weight 0.6 --out results/mmr_06
```

`config.yaml` holds `K`, `prior_weight`, `profile_weight`, and `diversity_weight`.
Each command creates a unique run directory under `--out`. A single run can also
compare weights using `--mmr-diversity-weights 0 0.3 0.6`. Complete archives include
the config, seeds, command, models, traces, metrics, figures and source snapshot.

Person A must confirm observable profiles, history policy, feedback units, and
article ordering. Current simulated feedback is utility after fatigue. Real-data
experiments also need the missing data-loading module and processed MIND files.

## Additional comparisons: BM25 and matrix factorisation

BM25 provides a lexical comparison to embedding-based content ranking. It scores
title-plus-abstract text using term frequency saturation and document length
normalization. It uses the Lucene positive-IDF convention:

```text
IDF(term) = log(1 + (N - document_frequency + 0.5) / (document_frequency + 0.5))
score(document, query) = sum over unique query terms of:
    IDF(term) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / mean_length))
```

Query terms come from optional visible initial query text and articles consumed
during simulation. Empty or out-of-vocabulary queries use learned popularity.
Missing or entirely empty article text raises an error. BM25 does not fabricate
text from categories, IDs, embeddings, or hidden user values. Tokenization is
deterministic lowercase alphanumeric matching; this is not a linguistic stemmer.

Defaults are `k1=1.2` and `b=0.75`. These are configurable, not tuned MIND results.
The formula follows [Elastic's BM25 explanation](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables).

Matrix factorisation uses an explicit-feedback model:

```text
predicted_utility(user, article) = global_mean + user_bias + article_bias
                                 + user_factors dot article_factors
```

Regularized alternating least squares fits only observed user–article utilities.
After each feedback batch it warm-starts from the previous factors and refits all
observed pairs. Missing feedback is not replaced with zero. Repeated observations
of a pair replace that pair's training target with the latest feedback, while
popularity still counts every observed read event.

Cold users use popularity. Items with no observations use a baseline rather than
random latent-factor predictions. The implementation follows the regularized
explicit-rating framework in [Koren, Bell and Volinsky (2009)](https://doi.org/10.1109/MC.2009.263).
Our target is the simulator's observed utility, not a calibrated click probability.

These defaults use eight factors, regularization one, and three ALS iterations
per feedback batch. Per-algorithm overrides are supported in `config.yaml`:

```yaml
recommenders:
  K: 10
  prior_weight: 1.0
  bm25:
    k1: 1.2
    b: 0.75
  mf:
    factors: 8
    regularization: 1.0
    iterations: 3
```

All constructor settings enter population metadata and report condition grouping.
MF factor arrays, biases, observed feedback, and popularity counts enter safe
checkpoints. BM25 rebuilds its index from archived article text and validates
the text fingerprint when loading a saved model.

## Why FAISS is deferred

[FAISS](https://faiss.ai/) provides vector-search indexes and related operations.
An exact inner-product index over normalized embeddings implements the same
ranking rule as the existing cosine scorer. Approximate indexes can change the
retrieved candidates. We have not added either backend as a separate behavioral
condition. A larger-catalogue performance benchmark should motivate that change.
