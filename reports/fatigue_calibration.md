# Empirical Calibration of Fatigue from Real MIND Click Logs

## 1. Research Question
How does reading consecutive articles in the same subcategory alter the empirical probability of continuing vs. switching topics, and how can this calibrate our agent simulation?

## 2. Streak Satiation Analysis

| Consecutive Reads ($k$) | Total Occurrences | P(Continue Same Subcategory) | P(Switch Topic) |
|---|---|---|---|
| 1 | 559,191 | 7.00% | **93.00%** |
| 2 | 39,120 | 15.50% | **84.50%** |
| 3 | 6,064 | 24.54% | **75.46%** |
| 4 | 1,488 | 29.77% | **70.23%** |
| 5 | 443 | 39.73% | **60.27%** |
| 6 | 176 | 43.18% | **56.82%** |
| 7 | 76 | 52.63% | **47.37%** |

*Finding*: Readers exhibit strong variety-seeking. After 1 article in a subcategory, readers switch to another topic **93.0%** of the time.

## 3. Discrete-Choice Fatigue Model
We model the log-odds of clicking article of subcategory $s$ as a function of baseline affinity and accumulated recent exposure stock $F_t(s) = \sum_{\tau < t} \delta^{t - \tau} \mathbb{I}(\text{click}_\tau = s)$:

$$\log \frac{P(\text{click} s)}{1 - P(\text{click} s)} = \beta_{\text{pref}} \cdot P_{\text{base}}(s) + \beta_{\text{fatigue}} \cdot F_t(s)$$

### Calibration Results:
- **Baseline Preference Weight ($\beta_{\text{pref}}$)**: `16.114`
- **Fatigue Penalty Coefficient ($\beta_{\text{fatigue}}$)**: `-0.590`
- **Decay Rate ($\delta$)**: `0.80`
- **Effective Relative Fatigue Strength**: $\phi \approx 0.037$

### Recommendation for Simulator:
- **Moderate / Calibrated Fatigue**: $\phi = 0.5$, $\text{decay} = 0.8$.
- **High Fatigue (Easily Bored)**: $\phi = 1.5$, $\text{decay} = 0.6$.
- **Low Fatigue (Explorer / Habitual)**: $\phi = 0.2$, $\text{decay} = 0.8$.
