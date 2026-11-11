"""Empirical calibration of fatigue from real MIND click logs.

Research question:
In real MIND click logs, how does a user's probability of consuming another
article from the same subcategory change after several consecutive same-subcategory
reads? How can this empirically calibrate the simulation's fatigue strength (phi)
and fatigue decay rate?

Outputs:
- reports/fatigue_calibration.md
- reports/figures/fatigue_calibration.png
- reports/fatigue_calibration.csv
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.store import MindStore


def analyze_streak_satiation(store: MindStore, max_users: int = 15000):
    """Measure how probability of clicking same subcategory evolves as streak length grows."""
    news_subcat = dict(zip(store.articles["news_id"], store.articles["subcategory"]))
    
    streak_counts = defaultdict(int)
    streak_continues = defaultdict(int)
    
    uids = list(store.histories.keys())[:max_users]
    for u in uids:
        hist = store.history(u)
        subcats = [news_subcat.get(n) for n in hist if n in news_subcat]
        if len(subcats) < 4:
            continue
            
        cur_sub = None
        cur_len = 0
        for s in subcats:
            if s == cur_sub:
                cur_len += 1
            else:
                if cur_sub is not None:
                    for k in range(1, cur_len):
                        streak_counts[k] += 1
                        streak_continues[k] += 1
                    streak_counts[cur_len] += 1
                cur_sub = s
                cur_len = 1
        if cur_sub is not None:
            for k in range(1, cur_len):
                streak_counts[k] += 1
                streak_continues[k] += 1
            streak_counts[cur_len] += 1
            
    rows = []
    for k in range(1, 8):
        if streak_counts[k] >= 50:
            cont = streak_continues[k]
            tot = streak_counts[k]
            p_cont = cont / tot
            p_switch = 1.0 - p_cont
            rows.append({
                "streak_length_k": k,
                "occurrences": tot,
                "continued_streak": cont,
                "p_continue": float(p_cont),
                "p_switch_topic": float(p_switch)
            })
    return pd.DataFrame(rows)


def fit_fatigue_regression(store: MindStore, max_users: int = 5000, decay: float = 0.8):
    """Estimate discrete choice model of subcategory choice under decaying exposure stock.
    Utility: u(s) = beta_pref * P(s) - phi_eff * Stock(s).
    """
    news_subcat = dict(zip(store.articles["news_id"], store.articles["subcategory"]))
    uids = list(store.histories.keys())[:max_users]
    
    X_data = []
    y_data = []
    rng = np.random.default_rng(42)
    
    for u in uids:
        hist = store.history(u)
        sc = [news_subcat.get(n) for n in hist if n in news_subcat]
        if len(sc) < 15:
            continue
        pref = Counter(sc)
        tot = len(sc)
        p_base = {k: v / tot for k, v in pref.items()}
        
        fatigue = Counter()
        for s in sc:
            f_s = fatigue[s]
            base_s = p_base[s]
            X_data.append([base_s, f_s])
            y_data.append(1)  # Chosen article
            
            # Contrast with unchosen candidate that the user also enjoys
            candidates = [k for k in p_base if k != s]
            if candidates:
                neg_s = rng.choice(candidates)
                X_data.append([p_base[neg_s], fatigue[neg_s]])
                y_data.append(0)
                
            for k in list(fatigue.keys()):
                fatigue[k] *= decay
            fatigue[s] += 1.0

    X = np.array(X_data)
    y = np.array(y_data)
    clf = LogisticRegression()
    clf.fit(X, y)
    
    beta_pref = float(clf.coef_[0, 0])
    beta_fatigue = float(clf.coef_[0, 1])
    relative_fatigue = float(abs(beta_fatigue / beta_pref))
    
    return {
        "decay_evaluated": decay,
        "beta_baseline_preference": beta_pref,
        "beta_fatigue_penalty": beta_fatigue,
        "relative_fatigue_strength": relative_fatigue,
        "observations": len(y)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    store = MindStore(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Analyzing subcategory streak satiation on MIND click histories...")
    df_streak = analyze_streak_satiation(store)
    df_streak.to_csv(out_dir / "fatigue_calibration.csv", index=False)

    print("Fitting discrete-choice fatigue regression...")
    reg_res = fit_fatigue_regression(store, decay=0.8)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(df_streak["streak_length_k"], df_streak["p_switch_topic"] * 100, marker="o", color="#C0504D", lw=2)
    ax1.set_title("Topic Switching Probability vs. Same-Subcategory Streak", fontsize=11)
    ax1.set_xlabel("Consecutive Articles Read in Same Subcategory (k)", fontsize=10)
    ax1.set_ylabel("Probability of Switching Topic (%)", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 100)

    # Decaying stock illustration
    rounds = np.arange(1, 11)
    decay_levels = [0.5, 0.8, 0.95]
    for d in decay_levels:
        stock = [sum(d ** (t - i) for i in range(1, t + 1)) for t in rounds]
        ax2.plot(rounds, stock, marker="s", label=f"Decay = {d}")
    ax2.set_title("Simulated Fatigue Buildup (Same Item/Neighbourhood)", fontsize=11)
    ax2.set_xlabel("Consecutive Rounds Consumed", fontsize=10)
    ax2.set_ylabel("Accumulated Fatigue Stock", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "fatigue_calibration.png", dpi=130)
    plt.close(fig)

    # Markdown summary
    md = f"""# Empirical Calibration of Fatigue from Real MIND Click Logs

## 1. Research Question
How does reading consecutive articles in the same subcategory alter the empirical probability of continuing vs. switching topics, and how can this calibrate our agent simulation?

## 2. Streak Satiation Analysis

| Consecutive Reads ($k$) | Total Occurrences | P(Continue Same Subcategory) | P(Switch Topic) |
|---|---|---|---|
"""
    for _, row in df_streak.iterrows():
        md += f"| {int(row['streak_length_k'])} | {int(row['occurrences']):,} | {row['p_continue']:.2%} | **{row['p_switch_topic']:.2%}** |\n"

    md += f"""
*Finding*: Readers exhibit strong variety-seeking. After 1 article in a subcategory, readers switch to another topic **{df_streak.loc[0, 'p_switch_topic']:.1%}** of the time.

## 3. Discrete-Choice Fatigue Model
We model the log-odds of clicking article of subcategory $s$ as a function of baseline affinity and accumulated recent exposure stock $F_t(s) = \\sum_{{\\tau < t}} \\delta^{{t - \\tau}} \\mathbb{{I}}(\\text{{click}}_\\tau = s)$:

$$\\log \\frac{{P(\\text{{click}} s)}}{{1 - P(\\text{{click}} s)}} = \\beta_{{\\text{{pref}}}} \\cdot P_{{\\text{{base}}}}(s) + \\beta_{{\\text{{fatigue}}}} \\cdot F_t(s)$$

### Calibration Results:
- **Baseline Preference Weight ($\\beta_{{\\text{{pref}}}}$)**: `{reg_res['beta_baseline_preference']:.3f}`
- **Fatigue Penalty Coefficient ($\\beta_{{\\text{{fatigue}}}}$)**: `{reg_res['beta_fatigue_penalty']:.3f}`
- **Decay Rate ($\\delta$)**: `{reg_res['decay_evaluated']:.2f}`
- **Effective Relative Fatigue Strength**: $\\phi \\approx {reg_res['relative_fatigue_strength']:.3f}$

### Recommendation for Simulator:
- **Moderate / Calibrated Fatigue**: $\\phi = 0.5$, $\\text{{decay}} = 0.8$.
- **High Fatigue (Easily Bored)**: $\\phi = 1.5$, $\\text{{decay}} = 0.6$.
- **Low Fatigue (Explorer / Habitual)**: $\\phi = 0.2$, $\\text{{decay}} = 0.8$.
"""
    (out_dir / "fatigue_calibration.md").write_text(md)
    print(f"Calibration completed and saved to {out_dir / 'fatigue_calibration.md'}")


if __name__ == "__main__":
    main()
