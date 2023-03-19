from lightfm import LightFM
import numpy as np



def build_lfm_model(config, data, data_description, early_stop_config=None, iterator=None):
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
            epochs=2,
            verbose=True)


    return model


def lightfm_scoring(model, data, data_description):
    """
    A standard scoring function adopted for use with LightFM in the item cold-start settings.
    It returns a 2D item-user array (i.e., a transposed matrix of interactions) corresponding
    to the predicted scores of user relevance to cold items.
    """
    dtype = 'i4'
    all_users = np.arange(data_description['n_users'], dtype=dtype)
    test_items = data_description['cold_items'].astype(dtype)
    item_index, user_index = np.meshgrid(test_items, all_users, copy=False)

    lfm_scores = model.predict(
        user_index.ravel(),
        item_index.ravel(),
        item_features = data_description['item_features']
    )
    scores = lfm_scores.reshape(len(test_items), len(all_users), order='F')
    return scores