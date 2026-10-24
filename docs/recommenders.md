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

Matrix factorisation is deferred until the minimum three systems work on real data.
These algorithms do not receive simulated true utilities or Bayesian beliefs.
The `ratings` field contains observed feedback only. These three algorithms count
reads, but do not weight content profiles by feedback values.

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
Existing calls without profiles continue to work.

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
python scripts/run_news.py --recommenders none random popularity content mmr --n-pop 20
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
