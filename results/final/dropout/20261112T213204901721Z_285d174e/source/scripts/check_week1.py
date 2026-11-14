import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.store import MindStore

store = MindStore()
print(f"Articles:   {len(store.articles)}")
print(f"Categories: {len(store.categories)} -> {store.categories}")
print(f"Users:      {len(store.histories)}")
print(f"Embedding matrix: {store.embeddings.shape}")

nid = store.articles["news_id"].iloc[0]
a = store.article(nid)
e = store.embedding(nid)
print(f"\nSample article {nid}: [{a['category']}] {a['title'][:80]}")
print(f"  popularity={a['popularity']:.3f}  embedding dim={e.shape[0]}  norm={np.linalg.norm(e):.3f}")

uid = store.user_ids()[0]
h = store.history(uid)
print(f"\nSample user {uid}: {len(h)} past clicks")
for n in h[-3:]:
    art = store.article(n)
    print(f"  [{art['category']}] {art['title'][:70]}")

assert abs(np.linalg.norm(e) - 1) < 1e-3, "embeddings should be normalised"
assert all(n in store.index for n in h), "history contains unknown articles"
assert 0 <= store.popularity.min() and store.popularity.max() <= 1
print("\nWeek 1 check passed.")
