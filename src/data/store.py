import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class MindStore:


    def __init__(self, config_path: str | Path = "config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.processed_dir = Path(cfg["paths"]["processed_dir"])


        self.articles = pd.read_csv(self.processed_dir / "articles.csv")
        self.index = {nid: idx for idx, nid in enumerate(self.articles["news_id"])}
        self.categories = sorted(self.articles["category"].dropna().unique().tolist())


        self.embeddings = np.load(self.processed_dir / "embeddings.npy")


        self.popularity = self.articles["popularity"].to_numpy(dtype=float)


        with open(self.processed_dir / "user_histories.json") as f:
            self.histories = json.load(f)

    def article(self, news_id: str) -> dict[str, Any]:
        idx = self.index[news_id]
        return self.articles.iloc[idx].to_dict()

    def embedding(self, news_id: str) -> np.ndarray:
        idx = self.index[news_id]
        return self.embeddings[idx]

    def history(self, user_id: str) -> list[str]:
        return self.histories.get(user_id, [])

    def user_ids(self) -> list[str]:
        return list(self.histories.keys())
