import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_news(raw_dir: Path) -> pd.DataFrame:
    cols = ["news_id", "category", "subcategory", "title", "abstract",
            "url", "title_entities", "abstract_entities"]
    train_news = pd.read_csv(raw_dir / "train" / "news.tsv", sep="\t", header=None,
                             names=cols, quoting=3, dtype=str)
    dev_news = pd.read_csv(raw_dir / "dev" / "news.tsv", sep="\t", header=None,
                           names=cols, quoting=3, dtype=str)

    combined = pd.concat([train_news, dev_news], ignore_index=True)
    combined = combined.drop_duplicates(subset=["news_id"], keep="first").reset_index(drop=True)
    combined["title"] = combined["title"].fillna("")
    combined["abstract"] = combined["abstract"].fillna("")
    combined["category"] = combined["category"].fillna("other")
    combined["subcategory"] = combined["subcategory"].fillna("other")
    return combined[["news_id", "category", "subcategory", "title", "abstract"]]


def compute_popularity(raw_dir: Path, articles: pd.DataFrame) -> np.ndarray:

    clicks = Counter()
    behaviors_path = raw_dir / "train" / "behaviors.tsv"
    with open(behaviors_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                impressions = parts[4].split()
                for imp in impressions:
                    if imp.endswith("-1"):
                        nid = imp[:-2]
                        clicks[nid] += 1

    counts = np.array([clicks.get(nid, 0) for nid in articles["news_id"]], dtype=float)
    log_counts = np.log1p(counts)
    max_log = log_counts.max()
    popularity = log_counts / max_log if max_log > 0 else np.zeros_like(log_counts)
    return popularity


def extract_user_histories(raw_dir: Path, valid_news_ids: set[str], min_history: int = 10) -> dict[str, list[str]]:

    behaviors_path = raw_dir / "train" / "behaviors.tsv"


    user_initial_history = {}
    user_session_clicks = defaultdict(list)
    rows = []

    with open(behaviors_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5:

                uid = parts[1]
                time_str = parts[2]
                hist_str = parts[3]
                imp_str = parts[4]
                rows.append((uid, time_str, hist_str, imp_str))


    for uid, time_str, hist_str, imp_str in rows:
        if uid not in user_initial_history:
            pre_clicks = [nid for nid in hist_str.split() if nid in valid_news_ids]
            user_initial_history[uid] = pre_clicks


        for imp in imp_str.split():
            if imp.endswith("-1"):
                nid = imp[:-2]
                if nid in valid_news_ids:
                    user_session_clicks[uid].append(nid)

    final_histories = {}
    for uid, pre in user_initial_history.items():

        full_seq = list(pre) + user_session_clicks[uid]
        if len(full_seq) >= min_history:
            final_histories[uid] = full_seq

    return final_histories


def compute_embeddings(articles: pd.DataFrame, model_name: str, batch_size: int = 256,
                       fake: bool = False) -> np.ndarray:
    if fake:
        print("Using fake embeddings...")
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((len(articles), 384)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    import os
    import torch
    num_threads = os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {model_name} (using {num_threads} torch threads)...")
    model = SentenceTransformer(model_name)

    texts = []
    for title, abstract in zip(articles["title"], articles["abstract"]):
        t = str(title).strip()
        a = str(abstract).strip()
        texts.append(f"{t}. {a}" if a else f"{t}.")

    print(f"Encoding {len(texts)} articles (batch_size={batch_size})...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                               normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fake-embeddings", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/4: Loading and deduplicating news articles...")
    articles = load_news(raw_dir)
    print(f"Total unique articles: {len(articles)}")

    print("Step 2/4: Computing article popularity from train impressions...")
    articles["popularity"] = compute_popularity(raw_dir, articles)

    print("Step 3/4: Computing embeddings...")
    embeddings = compute_embeddings(articles, cfg["embedding"]["model"],
                                    cfg["embedding"]["batch_size"],
                                    fake=args.fake_embeddings)
    np.save(processed_dir / "embeddings.npy", embeddings)
    articles.to_csv(processed_dir / "articles.csv", index=False)
    print("Saved articles.csv and embeddings.npy")

    print("Step 4/4: Extracting user histories...")
    valid_ids = set(articles["news_id"])
    histories = extract_user_histories(raw_dir, valid_ids, cfg["users"]["min_history"])
    print(f"Total qualified users (>= {cfg['users']['min_history']} clicks): {len(histories)}")
    with open(processed_dir / "user_histories.json", "w") as f:
        json.dump(histories, f)
    print("Saved user_histories.json")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
