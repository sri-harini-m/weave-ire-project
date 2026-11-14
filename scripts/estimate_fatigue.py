import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.store import MindStore

DECAYS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def user_sequences(store, min_len, max_users, seed):
    sub = dict(zip(store.articles["news_id"], store.articles["subcategory"]))
    uids = [u for u in store.user_ids() if len(store.history(u)) >= min_len]
    rng = np.random.default_rng(seed)
    if len(uids) > max_users:
        uids = list(rng.choice(uids, size=max_users, replace=False))
    return [[sub[n] for n in store.history(u) if n in sub] for u in uids]


def streak_counts(seqs, max_k):
    tot, cont = np.zeros(max_k + 1), np.zeros(max_k + 1)
    for s in seqs:
        run = 1
        for a, b in zip(s[:-1], s[1:]):
            k = min(run, max_k)
            tot[k] += 1
            if a == b:
                cont[k] += 1
                run += 1
            else:
                run = 1
    return tot, cont


def streak_table(seqs, max_k=6, n_shuffles=5, seed=0):
    tot, cont = streak_counts(seqs, max_k)
    rng = np.random.default_rng(seed)
    st, sc = np.zeros(max_k + 1), np.zeros(max_k + 1)
    for _ in range(n_shuffles):
        t, c = streak_counts([list(rng.permutation(s)) for s in seqs], max_k)
        st += t
        sc += c
    rows = []
    for k in range(1, max_k + 1):
        if tot[k] < 30 or st[k] < 30:
            continue
        p, q = cont[k] / tot[k], sc[k] / st[k]
        rows.append({"streak_k": k if k < max_k else f"{max_k}+", "transitions": int(tot[k]),
                     "p_continue_observed": p, "p_continue_shuffled": q,
                     "ratio_observed_to_shuffled": p / q if q > 0 else np.nan})
    return pd.DataFrame(rows)


def choice_events(seqs, decay, seed):

    rng = np.random.default_rng(seed)
    X, y = [], []
    for s in seqs:
        half = len(s) // 2
        base = Counter(s[:half])
        topics = list(base)
        if len(topics) < 2:
            continue
        pref = {t: c / half for t, c in base.items()}
        stock = Counter()
        for i, chosen in enumerate(s):
            if i >= half and chosen in pref:
                others = [t for t in topics if t != chosen]
                neg = others[rng.integers(len(others))]
                X += [[pref[chosen], stock[chosen]], [pref[neg], stock[neg]]]
                y += [1, 0]
            for t in list(stock):
                stock[t] *= decay
            stock[chosen] += 1.0
    return np.array(X, float), np.array(y, float)


def fit_logit(X, y, iters=50):

    Xd = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xd.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xd @ w))
        H = Xd.T @ (Xd * (p * (1 - p))[:, None]) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, Xd.T @ (y - p))
        w += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-Xd @ w))
    H = Xd.T @ (Xd * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return w, se


def loglik(w, X, y):
    p = np.clip(1 / (1 + np.exp(-np.column_stack([np.ones(len(X)), X]) @ w)), 1e-12, 1 - 1e-12)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--max-users", type=int, default=20000)
    ap.add_argument("--min-history", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    store = MindStore(args.config)
    seqs = user_sequences(store, args.min_history, args.max_users, args.seed)
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    streaks = streak_table(seqs, seed=args.seed)
    streaks.to_csv(out / "fatigue_streaks.csv", index=False)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(seqs))
    train = [seqs[i] for i in idx[: int(0.7 * len(seqs))]]
    test = [seqs[i] for i in idx[int(0.7 * len(seqs)):]]
    shuf = [list(rng.permutation(s)) for s in train]
    fits = []
    for d in DECAYS:
        Xtr, ytr = choice_events(train, d, args.seed)
        Xte, yte = choice_events(test, d, args.seed + 1)
        Xsh, ysh = choice_events(shuf, d, args.seed)
        w, se = fit_logit(Xtr, ytr)
        ws, ses = fit_logit(Xsh, ysh)
        eff, eff_se = w[2] - ws[2], float(np.hypot(se[2], ses[2]))
        fits.append({"decay": d, "heldout_loglik": loglik(w, Xte, yte),
                     "b_pref": w[1], "b_pref_se": se[1],
                     "b_fatigue_observed": w[2], "b_fatigue_observed_se": se[2],
                     "b_fatigue_shuffled": ws[2], "b_fatigue_shuffled_se": ses[2],
                     "order_effect": eff, "order_effect_se": eff_se,
                     "order_effect_odds_ratio": float(np.exp(eff)),
                     "train_events": len(ytr), "test_events": len(yte)})
    fit = pd.DataFrame(fits)
    fit.to_csv(out / "fatigue_decay_fit.csv", index=False)
    best = fit.loc[fit["heldout_loglik"].idxmax()]
    z = best["order_effect"] / best["order_effect_se"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ks = streaks["streak_k"].astype(str)
    ax[0].plot(ks, 100 * streaks["p_continue_observed"], "o-", label="observed order")
    ax[0].plot(ks, 100 * streaks["p_continue_shuffled"], "s--", label="shuffled order")
    ax[0].set(xlabel="consecutive same-subcategory clicks (k)",
              ylabel="P(next click same subcategory) %", title="Streak continuation")
    ax[0].legend()
    ax[1].plot(fit["decay"], fit["heldout_loglik"], "o-")
    ax[1].axvline(best["decay"], ls=":", c="grey")
    ax[1].set(xlabel="fatigue decay", ylabel="held-out log-likelihood", title="Choosing the decay")
    ax[2].errorbar(fit["decay"], fit["order_effect"], yerr=1.96 * fit["order_effect_se"], fmt="o-", capsize=3)
    ax[2].axhline(0, c="grey", lw=0.8)
    ax[2].set(xlabel="fatigue decay", ylabel="order effect (95% CI)",
              title="Recent exposure: observed minus shuffled")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fatigue_calibration.png", dpi=130)
    plt.close(fig)

    r1 = streaks.iloc[1] if len(streaks) > 1 else streaks.iloc[0]
    direction = ("below the shuffled baseline (satiation)" if r1["ratio_observed_to_shuffled"] < 1
                 else "above the shuffled baseline (momentum: readers binge a topic before moving on)")
    md = f"""# Fatigue estimated from MIND click histories

**Data.** {len(seqs):,} users with at least {args.min_history} clicks.

## 1. Streaks versus a shuffled baseline

Shuffling each user's history keeps their topic mix but removes order. Continuation
below the shuffled value indicates satiation, above it indicates momentum.

{streaks.round(4).to_string(index=False)}

After k = {r1['streak_k']} same-subcategory clicks, continuation is
{r1['ratio_observed_to_shuffled']:.2f}x the shuffled value, i.e. {direction}.
The raw switch rate after one click mostly reflects how many subcategories exist
and how spread out each user's interests are, not fatigue.

## 2. Choice model without look-ahead

Baseline preferences come from the first half of each history; clicks in the second
half are predicted. Decay chosen by held-out log-likelihood (70/30 user split).

{fit.round(4).to_string(index=False)}

**Best decay: {best['decay']:g}.** Order effect = {best['order_effect']:.3f}
(SE {best['order_effect_se']:.3f}, z = {z:.1f}): beyond what a user's topic mix
explains, one recent read in a subcategory multiplies the odds of choosing it again by
{best['order_effect_odds_ratio']:.3f}. {"This is satiation (fatigue)." if best['order_effect'] < 0 and z < -2 else "This is momentum (repeat reading)." if z > 2 else "No significant order effect."}

## 3. Use in the simulator

- The decay estimate ({best['decay']:g}) can inform `fatigue_decay`.
- The order effect is on a logit scale, not the simulator's utility scale, so it does
  not give phi directly. Its sign says whether fatigue (or momentum) is present in real
  reading; the phi values used (and the reader-type values) are modelling assumptions,
  and results should be reported across a range of phi, including phi = 0.
- Caveats: MIND histories are click logs from a live recommender, so order also
  reflects what was shown; subcategory is a coarse proxy for similarity.
"""
    (out / "fatigue_calibration.md").write_text(md)
    print(streaks.round(4).to_string(index=False))
    print(fit[["decay", "heldout_loglik", "b_fatigue_observed", "b_fatigue_shuffled",
               "order_effect", "order_effect_se"]].round(4).to_string(index=False))
    print(f"Best decay {best['decay']:g}; order effect {best['order_effect']:.3f} (z={z:.1f})")


if __name__ == "__main__":
    main()
