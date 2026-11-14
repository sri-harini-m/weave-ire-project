import numpy as np


def attach_user_pool(world, store, max_users: int = 5000, seed: int = 0,
                     fresh_readers: bool = True, min_history: int = 20,
                     true_history_size: int = 10):
    rng = np.random.default_rng(seed)
    all_uids = store.user_ids()
    uids = [u for u in all_uids if len(store.history(u)) >= min_history]
    total_qualified = len(uids)
    if not uids:

        uids = list(all_uids)
        total_qualified = len(uids)

    if len(uids) > max_users:
        uids = list(rng.choice(uids, size=max_users, replace=False))
    pos = {nid: j for j, nid in enumerate(world.ids)}
    dim = store.embeddings.shape[1]
    prior_profiles = np.zeros((len(uids), dim), np.float32)
    true_profiles = np.zeros((len(uids), dim), np.float32)
    history_lens = np.zeros(len(uids), dtype=int)
    seen = np.zeros((len(uids), world.N), bool)
    histories = []
    query_texts = []
    article_records = store.articles.set_index("news_id").to_dict("index")
    categories = np.unique(world.cats) if world.cats is not None else np.array([])
    category_index = {category: i for i, category in enumerate(categories)}
    category_profiles = np.zeros((len(uids), len(categories)))
    for k, u in enumerate(uids):
        hist = store.history(u)
        histories.append(list(hist))


        if len(hist) > true_history_size:
            prior_hist = hist[:-true_history_size]
            true_hist = hist[-true_history_size:]
        else:
            prior_hist = hist
            true_hist = hist

        history_lens[k] = len(prior_hist)
        prior_profiles[k] = store.embeddings[[store.index[n] for n in prior_hist]].mean(0)
        true_profiles[k] = store.embeddings[[store.index[n] for n in true_hist]].mean(0)

        text = []
        for news_id in prior_hist:
            article = article_records.get(news_id, {})
            category = article.get("category")
            if category in category_index:
                category_profiles[k, category_index[category]] += 1
            text.extend(article[key] for key in ("title", "abstract")
                        if isinstance(article.get(key), str))
        query_texts.append(" ".join(text))

        if not fresh_readers:
            for n in hist:
                if n in pos:
                    seen[k, pos[n]] = True

    world.user_pool = {
        "ids": np.array(uids),
        "profiles": prior_profiles,
        "prior_profiles": prior_profiles,
        "true_profiles": true_profiles,
        "history_lens": history_lens,
        "seen": seen,
        "histories": histories,
        "query_texts": query_texts,
        "category_profiles": category_profiles,
        "total_qualified_users": total_qualified,
        "min_history": min_history,
        "true_history_size": true_history_size,
    }
    return world


def sample_news_users(world, rng, U: int):

    pool = world.user_pool
    idx = rng.choice(len(pool["ids"]), size=U, replace=False)
    world.sampled_user_ids = pool["ids"][idx].tolist()
    world.sampled_histories = [pool["histories"][i] for i in idx] \
        if "histories" in pool else None
    world.sampled_query_texts = [pool["query_texts"][i] for i in idx] \
        if "query_texts" in pool else None
    world.sampled_true_profiles = pool["true_profiles"][idx] \
        if "true_profiles" in pool else None
    world.sampled_history_lens = pool["history_lens"][idx] \
        if "history_lens" in pool else None
    world.sampled_category_profiles = pool["category_profiles"][idx] \
        if "category_profiles" in pool else None
    return pool["profiles"][idx], pool["seen"][idx]
