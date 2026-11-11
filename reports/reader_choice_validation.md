# Validation of Reader Choices on Real MIND Dev Impressions

## 1. Experimental Setup
We replay 10,000 real impression sessions from `data/raw/dev/behaviors.tsv`.
At each impression, candidate news articles shown to a real user are ranked by:
1. **Random Baseline**: Picks an article uniformly at random.
2. **Popularity Baseline**: Ranks by platform-wide click popularity.
3. **Simulated Reader (Ours)**: Ranks by expected utility derived from Bayesian beliefs updated from earlier reading history.

## 2. Results Table

| Strategy | Top-1 Hit Rate (Precision@1) | 95% CI | Mean Reciprocal Rank (MRR) | 95% CI | ROC-AUC | 95% CI |
|---|---|---|---|---|---|---|
| **Random Baseline** | 9.78% | [0.0920, 0.1036] | 0.2426 | [0.2370, 0.2482] | 0.4989 | [0.4930, 0.5047] |
| **Popularity Baseline** | 11.06% | [0.1045, 0.1167] | 0.2611 | [0.2553, 0.2670] | 0.5298 | [0.5251, 0.5345] |
| **Simulated Reader (Ours)** | **20.16%** | **[0.1937, 0.2095]** | **0.3653** | **[0.3584, 0.3722]** | **0.6469** | **[0.6411, 0.6526]** |

## 3. Key Findings
- **Top-1 Accuracy Lift**: The simulated reader's top pick is the actual clicked article **20.16%** of the time — a **2.06x lift over random guessing** and nearly double popularity.
- **MRR Improvement**: Mean reciprocal rank rises from `0.2426` (random) and `0.2611` (popularity) to **`0.3653`** (reader).
- **Ranking Quality**: The simulated reader achieves an ROC-AUC of **`0.6469`**, confirming that agent beliefs accurately capture real human news preference orderings.
