import pandas as pd
import numpy as np

def topn_recommendations(scores, topn=10):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations


def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]


def calculate_novelty(train_interactions, recommendations, top_n):
    users = recommendations['user_id'].unique()
    n_users = train_interactions['user_id'].nunique()
    n_users_per_item = train_interactions.groupby('item_id')['user_id'].nunique()

    recommendations = recommendations.loc[recommendations['rank'] <= top_n].copy()
    recommendations['n_users_per_item'] = recommendations['item_id'].map(n_users_per_item)
    recommendations['n_users_per_item'] = recommendations['n_users_per_item'].fillna(1)
    recommendations['item_novelty'] = -np.log2(recommendations['n_users_per_item'] / n_users)

    item_novelties = recommendations[['user_id', 'rank', 'item_novelty']]

    miuf_at_k = item_novelties.loc[item_novelties['rank'] <= top_n, ['user_id', 'item_novelty']]
    miuf_at_k = miuf_at_k.groupby('user_id').agg('mean').squeeze()

    return miuf_at_k.reindex(users).mean()


def compute_metrics(train, test, recs, top_N):
    result = {}
    test_recs = test.set_index(['user_id', 'item_id']).join(recs.set_index(['user_id', 'item_id']))
    test_recs = test_recs.sort_values(by=['user_id', 'rank'])

    test_recs['users_item_count'] = test_recs.groupby(level='user_id')['rank'].transform(np.size)
    test_recs['reciprocal_rank'] = (1 / test_recs['rank']).fillna(0)
    test_recs['cumulative_rank'] = test_recs.groupby(level='user_id').cumcount() + 1
    test_recs['cumulative_rank'] = test_recs['cumulative_rank'] / test_recs['rank']

    users_count = test_recs.index.get_level_values('user_id').nunique()

    # Uncomment for Precision/Recall at k results

    #     for k in range(1, top_N + 1):
    #         hit_k = f'hit@{k}'
    #         test_recs[hit_k] = test_recs['rank'] <= k
    #         result[f'Precision@{k}'] = (test_recs[hit_k] / k).sum() / users_count
    #         result[f'Recall@{k}'] = (test_recs[hit_k] / test_recs['users_item_count']).sum() / users_count

    result[f'MAP@{top_N}'] = (test_recs['cumulative_rank'] / test_recs['users_item_count']).sum() / users_count
    result[f'Novelty@{top_N}'] = calculate_novelty(train, recs, top_N)

    return pd.Series(result)