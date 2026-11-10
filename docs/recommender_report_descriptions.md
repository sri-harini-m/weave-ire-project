# Recommender descriptions for the project report

This document describes the twelve recommender methods implemented in Weave.
Each paragraph can be adapted for the main report.
Descriptions cover implementation behavior, not claims that one method performs best.

## Shared simulation setup

Each round, a recommender selects up to K unread articles for each active reader.
The simulated reader then chooses an article using its beliefs and fatigue settings.
The `none` baseline allows full-catalogue browsing instead of selecting a slate.
Ordinary methods use public article information and observable interactions.
Only the `ideal` reference receives hidden true utilities.

When `reader_history_visible` is true, recommenders can also use initial MIND history information.
Otherwise, they learn reader-specific information from interactions during the simulation.
MIND supplies articles and historical clicks, while the simulator generates utility, fatigue, and boredom feedback.

## 1. No recommender (`none`)

This baseline allows each reader to browse every eligible article in the catalogue.
The reader chooses directly, without a restricted recommendation slate.
It provides a comparison for the effects of limiting article exposure through recommendations.
Changing K does not restrict this baseline.

## 2. Random recommendation (`random`)

This method randomly selects up to K eligible articles for each active reader.
It uses no preference model or popularity score.
A seeded random generator makes its selections reproducible.
It provides a comparison against rankings that use article or reader information.

## 3. Popularity recommendation (`popularity`)

This method ranks articles using shared read counts across simulated readers.
An optional public popularity prior initializes these counts.
Every observed read adds one count, regardless of whether its utility is positive or negative.
The method gives readers the same article scores, although their eligible articles can differ.

## 4. Content-based recommendation (`content`)

This method ranks articles by cosine similarity between article embeddings and a reader profile.
The profile combines any visible initial profile with embeddings of articles consumed during the simulation.
Readers without a usable profile or reading history receive popularity-based rankings.
Reading history updates the profile without weighting its articles by utility feedback.

## 5. Maximal Marginal Relevance (`mmr`)

This method builds a slate by combining content relevance with a penalty for similarity between selected articles.
It first selects the most relevant article, then repeatedly penalizes candidates similar to articles already selected.
The implemented score is `relevance - diversity_weight * maximum_similarity_to_selected_articles`.
A weight of zero matches content ranking; larger weights increase the similarity penalty.
This diversity-penalty convention differs from MMR formulations where lambda weights relevance.

## 6. BM25 lexical recommendation (`bm25`)

This method ranks article titles and abstracts using BM25 keyword relevance.
Its query combines any visible initial history text with text from articles consumed during the simulation.
BM25 accounts for term rarity, repeated terms, and document length.
Queries without matching vocabulary use popularity rankings instead.

## 7. Matrix factorization (`mf`)

This method predicts observed utility using user factors, article factors, and user and article biases.
Regularized alternating least squares refits the model after each feedback batch using accumulated observations.
Training includes observed zero and negative utilities; missing observations remain excluded.
New readers use popularity rankings, while unseen articles receive fitted baseline scores instead of random factor predictions.
Its training target is utility after fatigue, not click probability or hidden true preference.

## 8. Ideal reference (`ideal`)

This privileged reference ranks eligible articles using each reader's true simulated utility before fatigue.
It adapts the ideal-reference idea described by Chaney, Stewart, and Engelhardt in Section 5.1 of their paper.
Ordinary recommenders cannot access these true utilities.
Readers still choose within the slate, so this reference does not guarantee maximum realized utility after fatigue.

## 9. Calibrated recommendation (`calibrated`)

This method combines content relevance with matching the slate's category distribution to a reader's known topic mix.
The target mix combines visible historical category counts with reads observed during the simulation.
Without known category history, it uses the public catalogue's category distribution.
Greedy selection balances relevance against a smoothed Kullback–Leibler divergence penalty.
Calibration follows known reading proportions; it does not require equal coverage of every topic.

## 10. Fatigue-aware popularity (`fatigue_popularity`)

This variant reduces an article's popularity-count increment when a reader reports boredom after consuming it.
Each read contributes `1 - fatigue_weight * boredom`, where boredom is zero or one.
It uses observable boredom reports rather than hidden fatigue values or unobserved preferences.
A fatigue weight of zero reproduces ordinary popularity updates.

## 11. Fatigue-aware matrix factorization (`fatigue_mf`)

This variant trains matrix factorization on `observed_net_utility - fatigue_weight * boredom`.
It also reduces popularity-count increments for bored reads, affecting the fallback for readers without observations.
The boredom penalty supplements utility that already includes fatigue; it does not recover utility before fatigue.
A fatigue weight of zero reproduces ordinary MF behavior when other settings and random seeds match.

## 12. Fatigue-aware content recommendation (`fatigue_content`)

This variant subtracts a recent-category consumption penalty from ordinary content relevance.
Each read adds category load, while a bored read adds extra load.
The load decays over rounds, reducing penalties for categories that have not been consumed recently.
The method uses observed categories, reads, and boredom reports without accessing the reader's hidden fatigue state.
A fatigue weight of zero reproduces ordinary content rankings.

## Feedback and interpretation

The simulator models boredom as a binary self-report when experienced fatigue exceeds the configured threshold.
This is simulated feedback, not a boredom label supplied by MIND.
The fatigue-aware methods also record explicit departures without assigning dislikes to unread articles.
The comparisons measure category diversity, article overlap, utility, fatigue, consumption, and other saved metrics.
Improved diversity does not automatically imply improved utility or lower fatigue.

The completed MIND-based pilot covers 320 conditions and 640 population runs.
It uses two populations per condition and therefore provides limited precision.
The larger study configuration is available, but that full study has not been run.
See the [pilot report](../reports/person_b_recommendation_pilot/README.md) for results and verification evidence.

## Implementation and references

- [Recommender source](../src/sim/recommenders.py) contains all twelve methods.
- [Study notes](recommendation_study.md) describe controls, equations, archive policies, and implementation limits.
- [Chaney et al.: algorithmic confounding and the ideal reference](https://arxiv.org/abs/1710.11214).
- [Steck: calibrated recommendations](https://doi.org/10.1145/3240323.3240372).
- [Koren, Bell, and Volinsky: matrix factorization techniques](https://doi.org/10.1109/MC.2009.263).

The ideal and calibrated methods adapt these reference ideas to the existing simulator.
The project does not reproduce those papers' complete experiments.
