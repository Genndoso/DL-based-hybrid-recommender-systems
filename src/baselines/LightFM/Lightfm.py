from lightfm import LightFM
import numpy as np


def build_lfm_model(config, data, data_description, n_epochs = 20):
    """
    Builds a LightFM model using the given configuration, data and data description.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration for the model. It must contain the following keys:
        'num_components', 'max_sampled', 'loss', 'learning_schedule', 'user_alpha' and 'item_alpha'.
    data : sparse matrix of interactions in COO format of shape (n_users, n_items)
        The training data.
    data_description : dict
        A dictionary containing information about the data. It must contain the following keys:
        'interactions', 'user_features' and 'item_features'.
    early_stop_config : dict, optional (default=None)
        A dictionary containing early stopping configuration. If not provided, default values will be used.

    Returns
    -------
    model : LightFM object The trained LightFM model.
    """
    # the model
    model = LightFM(
        no_components=config['no_components'],
        loss=config['loss'],
        learning_schedule=config['learning_schedule'],
        # learning_rate=
        user_alpha=config['user_alpha'],
        item_alpha=config['item_alpha'],
        max_sampled=config['max_sampled'],
        # random_state =
    )
    model.fit(data,
            user_features=data_description['user_features'],
            item_features=data_description['item_features'],
            epochs=n_epochs,
            verbose=True)


    return model


def lightfm_scoring(lfm, preds, data_description, users_to_val=10000):
    dtype = 'i4'
    all_users = np.arange(data_description['n_users_train'], dtype=dtype)
    test_items = np.arange(data_description['n_items']).astype(dtype)
    item_index, user_index = np.meshgrid(test_items, all_users, copy=False)
    all_items = np.arange(data_description['n_items'], dtype=dtype)

    for i in range(len(all_users)):
        if i % 1000 == 0:
            print(i)
        if i == users_to_val:
            break
        score = lfm.predict(user_index[i].ravel(), item_ids=all_items.ravel(),
                            user_features=data_description['user_features'],
                            item_features=data_description['item_features'],
                            num_threads=4)
        scores = np.expand_dims(score, axis=0)
        scores_topn = topn_recommendations(scores, topn)
        preds[i, :] = scores_topn

    hr_full = []
    mrr_full = []
    cov_full = []
    ndcg_full = []
    for i in holdout.user_id.sort_values():
        if i % 1000 == 0:
            print(i)
        if i == users_to_val:
            break
        predictions = preds[i]

        hr, mrr, cov, ndcg = user_evaluate(i, preds, holdout)
        hr_full.append(hr)
        mrr_full.append(mrr)
        cov_full.append(cov)
        ndcg_full.append(ndcg)

    return np.array(hr_full), np.array(mrr_full), np.array(cov_full)


def lightfm_eval(preds,holdout,data_description, users_to_eval = 10000):

    hr_full = []
    mrr_full = []
    cov_full = []
    ndcg_full = []
    for count,i in enumerate(holdout.user_id.sort_values()):
        if count % 100 == 0:
            print(count,i)
        if count == users_to_eval:
            break
       # predictions = preds[i]

        hr, mrr, cov, ndcg = user_evaluate(i,preds, holdout, data_description)
        hr_full.append(hr)
        mrr_full.append(mrr)
        cov_full.append(cov)
        ndcg_full.append(ndcg)
    return np.array(hr_full), np.array(mrr_full), np.array(cov_full),  np.array(ndcg_full)