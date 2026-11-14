import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.store import MindStore


def evaluate_dev_impressions(store: MindStore, dev_behaviors_path: Path, max_impressions: int = 10000,
                             seed: int = 42):
    rng = np.random.default_rng(seed)

    auc_reader = []
    auc_pop = []
    auc_rand = []

    evaluated = 0
    with open(dev_behaviors_path, "r", encoding="utf-8") as f:
        for line in f:
            if evaluated >= max_impressions:
                break
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue

            uid, hist_str, imp_str = parts[1], parts[3], parts[4]


            hist_nids = [nid for nid in hist_str.split() if nid in store.index]
            if not hist_nids:
                continue


            candidates = []
            labels = []
            for item in imp_str.split():
                if "-" not in item:
                    continue
                nid, click = item.rsplit("-", 1)
                if nid in store.index:
                    candidates.append(nid)
                    labels.append(int(click))

            labels = np.array(labels)

            if len(labels) < 2 or labels.sum() == 0 or labels.sum() == len(labels):
                continue


            c_idx = [store.index[n] for n in candidates]
            E_cand = store.embeddings[c_idx]
            pop_cand = store.popularity[c_idx]


            h_idx = [store.index[n] for n in hist_nids]
            profile = store.embeddings[h_idx].mean(axis=0)
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile /= norm


            scores_reader = E_cand @ profile
            scores_pop = pop_cand
            scores_rand = rng.random(len(labels))

            try:
                auc_reader.append(roc_auc_score(labels, scores_reader))
                auc_pop.append(roc_auc_score(labels, scores_pop))
                auc_rand.append(roc_auc_score(labels, scores_rand))
                evaluated += 1
            except Exception:
                continue

    results = {
        "Model": ["Random", "Popularity", "Simulated Reader Beliefs (Ours)"],
        "Mean_AUC": [float(np.mean(auc_rand)), float(np.mean(auc_pop)), float(np.mean(auc_reader))],
        "Std_Err": [float(np.std(auc_rand) / np.sqrt(len(auc_rand))),
                    float(np.std(auc_pop) / np.sqrt(len(auc_pop))),
                    float(np.std(auc_reader) / np.sqrt(len(auc_reader)))],
        "CI_95_Lower": [float(np.mean(auc_rand) - 1.96 * np.std(auc_rand) / np.sqrt(len(auc_rand))),
                        float(np.mean(auc_pop) - 1.96 * np.std(auc_pop) / np.sqrt(len(auc_pop))),
                        float(np.mean(auc_reader) - 1.96 * np.std(auc_reader) / np.sqrt(len(auc_reader)))],
        "CI_95_Upper": [float(np.mean(auc_rand) + 1.96 * np.std(auc_rand) / np.sqrt(len(auc_rand))),
                        float(np.mean(auc_pop) + 1.96 * np.std(auc_pop) / np.sqrt(len(auc_pop))),
                        float(np.mean(auc_reader) + 1.96 * np.std(auc_reader) / np.sqrt(len(auc_reader)))],
        "Evaluated_Impressions": [evaluated, evaluated, evaluated]
    }
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-impressions", type=int, default=10000)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_dir = Path(cfg["paths"]["raw_dir"])
    dev_behaviors = raw_dir / "dev" / "behaviors.tsv"

    print(f"Loading MindStore from {args.config}...")
    store = MindStore(args.config)

    print(f"Evaluating reader beliefs on up to {args.max_impressions} dev impressions...")
    df = evaluate_dev_impressions(store, dev_behaviors, max_impressions=args.max_impressions)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "reader_validation.csv", index=False)

    print("\n--- Reader Validation on MIND Dev Data ---")
    print(df.to_string(index=False))


    md_content = "# Validation of Simulated Readers on MIND Dev Data\n\n"
    md_content += "Evaluated across real user impressions in `data/raw/dev/behaviors.tsv`:\n\n"
    md_content += df.to_markdown(index=False) + "\n"
    (out_dir / "reader_validation.md").write_text(md_content)
    print(f"\nSaved results to {out_dir / 'reader_validation.csv'} and {out_dir / 'reader_validation.md'}")


if __name__ == "__main__":
    main()
