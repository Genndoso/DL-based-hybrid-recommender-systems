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