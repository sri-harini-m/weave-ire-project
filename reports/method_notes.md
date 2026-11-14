# Simulation Methodology, Calibration, and Validation Notes

This document provides complete documentation of the simulation design, mathematical equations, empirical calibration, offline validation, and reader archetype definitions. You can directly draw from these sections when writing your final report.

---

## 1. Motivation & Scientific Rationale

### The Limitations of Standard Synthetic ABMs
In the original model of Aridor, Gonçalves, & Sikdar (2020), readers are placed on a 1D circle where:
1. True item values $X_{i,n} = V_{i,n} + \beta V_n$ are centered on prior beliefs $\bar{v}_i$, with idiosyncratic tastes drawn as pure Gaussian noise: $V_i \sim \mathcal{N}(\bar{v}_i, \sigma_i^2 C)$.
2. The discrepancy between what a reader *believes* they enjoy and what they *actually* experience is treated as ungrounded white noise.
3. Every reader has identical, static taste uncertainty, irrespective of how many articles they have ever read.

### Our Two Key Real-World Groundings
To ground the simulation in real human reading behavior on the MIND news dataset, we introduce two methodological innovations:

1. **Temporal History Split (Empirical Taste Drift)**:
   - For every real MIND user, we partition their reading history chronologically into two disjoint subsets:
     - **Prior History ($H_{\text{prior}}$)**: All earlier clicks except the last 10, defining the reader's *expected tastes* (prior beliefs $\bar{v}_i$).
     - **Recent History ($H_{\text{true}}$)**: The fixed 10 most recent clicks, defining the reader's *true current tastes* ($v_i^*$).
   - The divergence $\delta_i = v_i^* - \bar{v}_i$ reflects **empirical interest drift over time**, rather than arbitrary Gaussian noise. When a reader consumes an article, positive or negative surprise arises from real changes in personal reading habits.
   - **Sample filtering**: Requiring $|H_{\text{prior}}| \ge 10$ and $|H_{\text{true}}| = 10$ (total history $\ge 20$) yields **18,215 qualified users** (55.2% of all active users in MIND-small), providing a rich pool of realistic preference pairs.

2. **History-Dependent Taste Uncertainty ($n_0$)**:
   - A reader with 100 historical reads has well-crystallized tastes, whereas a casual reader with 12 reads is uncertain about their preferences.
   - Grounded in conjugate Normal Bayesian updating with prior strength $n_0$, each reader's initial taste variance scales inversely with their historical observation count $n_i = |H_{\text{prior}}|$:
     $$s_i^2 = s_0^2 \cdot \frac{n_0}{n_0 + n_i}$$
   - Under CARA risk aversion ($\text{CE}_{in} = \mu_{in} - \frac{\gamma_i}{2} \text{var}_{in}$), casual readers experience greater uncertainty penalties, making them more hesitant to explore new categories and creating an emergent, realistic **cold-start filter bubble**.

---

## 2. Mathematical Formulation

### 2.1 Latent Space & Correlation Matrix
From 384-dimensional SentenceTransformer (`all-MiniLM-L6-v2`) embeddings, articles are projected via PCA to $d=8$ dimensions to mitigate distance concentration:
- Normalized distance: $D_{nm} = \frac{\|z_n - z_m\|_2}{\max_{j,k} \|z_j - z_k\|_2} \in [0, 1]$.
- Neighborhood unit: $\text{unit} = \text{median}_{n} \min_{m \ne n} D_{nm}$.
- Correlation matrix: $C_{nm} = \rho^{D_{nm} / \text{unit}}$, with $C_{nn} = 1.0$.

### 2.2 Expected vs. True Preferences
Let $E \in \mathbb{R}^{N \times 384}$ be the L2-normalized article embeddings, and let $P_{\text{prior}}, P_{\text{true}} \in \mathbb{R}^{U \times 384}$ be the mean embeddings of the reader's prior and recent click histories:
- **Prior belief mean**:
  $$\bar{v}_{i} = \sigma_{\text{bar}} \cdot \text{zscore}(P_{\text{prior}, i} E^T)$$
- **True taste center**:
  $$v_{i}^* = \sigma_{\text{bar}} \cdot \text{zscore}(P_{\text{true}, i} E^T)$$
- **True realized utility**:
  $$V_i = v_i^* + \sigma_i \cdot (\epsilon_i L^T), \quad \epsilon_i \sim \mathcal{N}(0, I_N)$$
  $$X_i = V_i + \beta V, \quad V \sim \mathcal{N}(0, \sigma^2 C)$$
- **Initial beliefs**:
  $$\mu_{0, i} = \begin{cases} \bar{v}_i & \text{under info='none'} \\ \bar{v}_i + \beta V & \text{under info='rec'} \\ X_i & \text{under info='oracle'} \end{cases}$$
  $$s_{0, i}^2 = \begin{cases} (\sigma_i^2 + \beta^2 \sigma^2) \cdot \frac{n_0}{n_0 + n_i} & \text{under info='none'} \\ \sigma_i^2 \cdot \frac{n_0}{n_0 + n_i} & \text{under info='rec'} \\ 0 & \text{under info='oracle'} \end{cases}$$

### 2.3 Bayesian Updating & Learning Spillovers
After consuming article set $S$ and observing realized values $X_{i, S}$:
$$\mu_{i} = \mu_{0, i} + C_{:, S} (C_{SS} + \epsilon I)^{-1} (X_{i, S} - \mu_{0, i, S})$$
$$\text{var}_{i} = s_{0, i}^2 \cdot \left(1 - \text{diag}\left(C_{:, S} (C_{SS} + \epsilon I)^{-1} C_{S, :}\right)\right)$$

### 2.4 Dynamic Fatigue Stock
Fatigue accumulates across similar articles and decays over rounds:
$$K_f(n, m) = \rho_f^{D_{nm} / \text{unit}}$$
$$F_{i, t} = \delta_i \cdot F_{i, t-1} + K_f(:, n_{t-1})$$
The certainty equivalent net of anticipated fatigue is:
$$\text{Score}_{i, n, t} = \mu_{i, n, t} - \frac{\gamma_i}{2} \text{var}_{i, n, t} - \phi_i F_{i, n, t}$$
The realized utility experienced is:
$$R_{i, n, t} = X_{i, n} - \phi_i F_{i, n, t}$$

---

## 3. Heterogeneous Reader Archetypes

Instead of a single uniform population, the simulator supports 4 distinct behavioral archetypes:

| Archetype | Risk Aversion ($\gamma$) | Fatigue Sensitivity ($\phi$) | Decay Rate ($\delta$) | Behavioral Description |
|---|---|---|---|---|
| **Cautious** | $2.0$ | $0.5$ | $0.8$ | Highly averse to taste uncertainty; heavily penalizes unexplored topics; strong tendency to form tight filter bubbles. |
| **Explorer** | $0.0$ | $0.2$ | $0.8$ | Risk-neutral; selects purely on expected mean utility; explores broadly across categories; low fatigue. |
| **Easily Bored** | $0.5$ | $1.5$ | $0.6$ | Moderate risk aversion but extreme fatigue penalty; rapidly switches topics after 1-2 reads; drops out early if feed lacks novelty. |
| **Average** | $1.0$ | $0.5$ | $0.8$ | Balanced benchmark reader; steady learning and moderate fatigue buildup. |

Continuous range sampling is also supported: $\gamma \sim U(0, 2)$, $\phi \sim U(0.1, 1.5)$, $\delta \sim U(0.5, 0.9)$.

---

## 4. Fatigue in Real MIND Reading

**Script:** `scripts/estimate_fatigue.py` → `reports/final/validation/fatigue_calibration.md`,
`fatigue_streaks.csv`, `fatigue_decay_fit.csv`, `figures/fatigue_calibration.png`.
Fill the numbers below from those files after the final run.

### 4.1 Streaks against a shuffled baseline
Raw continuation probabilities (e.g. "93% switch topic after one read") mostly reflect how
many subcategories exist and how spread out each user's interests are, so on their own they
are not evidence of fatigue. We therefore compare P(next click in the same subcategory | k
consecutive same-subcategory clicks) with the same quantity after shuffling each user's
history (same topic mix, no order). Observed below shuffled = satiation; above = momentum.
In the earlier analysis, continuation *rose* with streak length (7% → 53%), which points to
momentum or topic loyalty rather than fatigue; the shuffled comparison tests this properly.

### 4.2 Choice model without look-ahead
Baseline topic preference is computed from the first half of each user's history, and clicks
in the second half are predicted from preference and a decaying exposure stock
F_t(s) = Σ_{τ<t} δ^{t-τ} 1[click_τ ∈ s]. The decay δ is chosen by held-out log-likelihood
over a grid (not assumed). Because the exposure stock also partly tracks how often a user
reads a topic, the coefficient is biased upwards even when order carries no information; we
report the **order effect** = coefficient on real order minus coefficient on shuffled order.

### 4.3 How this is used in the simulator
- The estimated decay can inform `fatigue_decay`.
- The order effect is on a logit scale, not the simulator's utility scale, so it does not
  give φ directly. Its sign tells us whether real readers show satiation or momentum.
  The φ values (0, 0.5, 1 and the reader-type values) are modelling assumptions, so results
  are reported across φ, including φ = 0.
- Caveats: MIND logs come from a live recommender (order partly reflects what was shown),
  and subcategory is a coarse proxy for similarity.

---

## 5. Validation of the Reader Choice Rule on Real Impressions

**Script:** `scripts/validate_reader_choices.py` → `reports/final/validation/reader_choice_validation.md`,
`reader_choice_validation.csv`, `reader_choice_validation_grid.csv`.

The earlier version of this check ranked shown articles only by cosine similarity to the
user's mean history embedding. That validates the *starting beliefs*, not the choice rule
(it is the same check as the mid-submission, AUC 0.647).

The final version scores each shown article with the simulator's own choice rule:
score = mean − γ/2 · variance − φ · fatigue, using the simulator's correlation and fatigue
kernels, where the user's most recent clicks (up to 30) are treated as already read.
Past reads are assumed to bring no surprise (their values are not observable in MIND), so
they change the variance and fatigue terms but not the mean. γ = φ = 0 reproduces the plain
similarity ranking. We report AUC, top-1 hit rate and MRR with 95% CIs for random,
popularity, plain similarity, the four reader types and the best-fitting (γ, φ) from a grid.

Users are split into two halves by user id. The (γ, φ) grid is searched on the fit half
only, and every scorer in the main table, including the best fit, is scored on the
held-out test half. The script also reports the paired AUC gain of the best fit over plain
similarity on the test half, with a 95% CI.

How to read it: if that gain's CI includes zero, adding variance and fatigue (γ, φ > 0) does
not improve on plain similarity for predicting single clicks. The extra terms still matter
for how reading evolves over a session, which single impressions cannot show.

---

## 6. Suggested Text for the Report

> *"We model news readers with the Bayesian-learning agent model of Aridor et al. (2020),
> extended to real articles from the Microsoft News Dataset (MIND). Each reader's click
> history is split chronologically: earlier clicks define their prior beliefs and their 10
> most recent clicks define their current true tastes, so the gap between belief and truth
> reflects real drift in interests. Prior uncertainty shrinks with history length,
> s_i² = s_0² · n_0 / (n_0 + n_i). Readers differ in risk aversion and fatigue sensitivity
> (cautious, explorer, easily bored, average). Fatigue accumulates with exposure to
> similar articles and decays over rounds; [report the shuffled-baseline result from
> fatigue_calibration.md here]. Scoring real MIND dev impressions with the simulator's choice
> rule gives a held-out ROC-AUC of [X] (popularity [Y], random [Z]); [give the held-out gain
> of the best (γ, φ) over plain similarity and say whether its CI includes zero]."*
