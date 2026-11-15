# Weave: filter bubbles and fatigue in news recommendation

Person B's recommender input contract, defaults, and checks are in
[docs/recommenders.md](docs/recommenders.md).
Saved report artifacts and offline rebuild commands are in
[docs/report_artifacts.md](docs/report_artifacts.md).
The K/history/MMR sweeps, ideal reference, calibrated ranking and explicit fatigue
feedback are documented in [docs/recommendation_study.md](docs/recommendation_study.md).

Agent-based simulation (Aridor, Goncalves & Sikdar, "Deconstructing the Filter
Bubble") extended to news (MIND-small) with user fatigue, used to evaluate
filter bubbles, echo chambers and fatigue under different recommenders.

## Repo layout

```
config.yaml              all settings (paths, embedding, simulation grids)
requirements.txt
scripts/
  download_mind.py       downloads + unzips MIND-small into data/raw/
  check_week1.py         the Week 1 "done when" test
src/data/
  prepare.py             builds articles, popularity, embeddings, user histories
  store.py               MindStore: the one way everyone loads the data
data/                    (git-ignored) raw/ and processed/
```

## Week 1 setup

**1. Create the environment** (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Download MIND-small** (about 100 MB; you accept the
[MIND license](https://msnews.github.io/) by downloading)

```bash
python scripts/download_mind.py
```

You should now have `data/raw/train/` and `data/raw/dev/`, each containing
`news.tsv` and `behaviors.tsv`.

*If the download fails:* get `MINDsmall_train.zip` and `MINDsmall_dev.zip`
from https://msnews.github.io/ (or the Hugging Face mirror
`loicmagne/mind_small`), unzip them into `data/raw/train/` and
`data/raw/dev/`, and continue.

**3. Prepare the data**

```bash
python -m src.data.prepare
```

This takes about 5–15 minutes on a laptop CPU (mostly embedding) and much less on a GPU.
To test the pipeline quickly first, run with `--fake-embeddings`.

It writes to `data/processed/`:

| File | Contents |
|---|---|
| `articles.csv` | news_id, category, subcategory, title, abstract, popularity (0–1) |
| `embeddings.npy` | one normalized 384-dim vector per article, same row order as `articles.csv` |
| `user_histories.json` | `{user_id: [news_id, ...]}`, oldest first, users with ≥ 10 clicks |

**4. Run the check**

```bash
python scripts/check_week1.py
```

If it ends with `Week 1 check passed.`, Week 1 is done.

## Using the data in code

```python
from src.data.store import MindStore
store = MindStore()
store.article("N55528")     # {'category': ..., 'title': ..., 'popularity': ...}
store.embedding("N55528")   # numpy vector (384,)
store.history("U13740")     # list of news_ids
store.embeddings            # full matrix, for fast scoring
store.popularity            # popularity array, same order
```

## Design decisions (worth noting in the report)

- **Catalog** = all articles from train + dev, de-duplicated.
- **Popularity** = log(1 + clicks in *train* impressions), scaled to [0, 1].
  Only train is used, so dev stays clean for offline evaluation later.
- **User history** = the user's pre-period history, followed by their clicks
  during the train week, in time order.
- **Embeddings** = `all-MiniLM-L6-v2` on "title. abstract", L2-normalized,
  so cosine similarity is just a dot product.

## The simulation (Aridor model)

```
src/sim/
  params.py          every model parameter (paper names + our fatigue knobs)
  world.py           item space: circle (paper) or MIND articles (news)
  users.py           users' true values, priors, Bayesian beliefs
  simulate.py        the round-by-round loop
  recommenders.py    plug-in interface for recommendation systems
  metrics.py         locality, diversity, welfare, homogeneity, fatigue, entropy
  experiment.py      parameter grids, many populations, summaries with CIs
  news_users.py      real MIND users as simulated agents
scripts/
  reproduce_aridor.py   step 1: reproduce the paper on the circle
  run_news.py           step 2+: same model on news, with fatigue
tests/test_sim.py       unit tests (Bayesian update vs brute force, etc.)
```

Run, from the repo root:

```bash
python -m pytest -q                                   # 1. tests (seconds)
python scripts/reproduce_aridor.py --workers 4        # 2. reproduce paper (~5-10 min)
python scripts/run_news.py --workers 4                # 3. news + fatigue (needs Week 1 data)
python scripts/run_news.py --recommenders none random # 4. compare recommenders
```

Results go to `results/`. Check `results/aridor_repro/checks.txt`: all lines
should say PASS before trusting the news results.

Decisions the paper leaves open (state these in the report):
- prior spread sigma_bar = 1, and sigma_i = sigma
- under "rec", the recommender reveals V; users still choose themselves
- news distance = embedding distance after PCA to 8 dims; neighbour
  correlation = rho, with the "neighbour" distance = median nearest-neighbour distance
- news priors = z-scored similarity to the user's real MIND click history

## Data is not committed

`data/` is in `.gitignore`. Each teammate runs steps 2–3 once on their own machine.
