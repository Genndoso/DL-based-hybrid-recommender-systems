import pandas as pd
from scipy.sparse import csr_matrix


def preprocessing(interactions_df, users_df_ohe, items_df_ohe, cold_users_split=5, itemid='last_watch_dt'):
    interactions_df = interactions_df[interactions_df.user_id.isin(users_df_ohe.user_id.unique())]
    interactions_df['last_watch_dt_ts'] = interactions_df['last_watch_dt'].apply(lambda x: int(x.timestamp()))
    num_interaction_pu = interactions_df.groupby('user_id')['item_id'].count().sort_values(ascending=False)
    # get cold_users
    cold_users = num_interaction_pu.loc[(num_interaction_pu < 5) & (num_interaction_pu > 2)].index

    # warm_users_history
    warm_users_history = interactions_df[~interactions_df.user_id.isin(cold_users)]

    # cold_users_history
    cold_users_history = interactions_df[interactions_df.user_id.isin(cold_users)]

    # standard scenario train/holdout split
    training, holdout = leave_last_out(warm_users_history, userid='user_id', timeid=itemid)

    train_val, data_index_train = transform_indices(training, 'user_id', 'item_id')
    holdout_val = reindex_data(holdout, data_index_train, fields="items")

    # cold_start_scenario train/holdout split
    training, holdout = leave_last_out(cold_users_history, userid='user_id', timeid=itemid)

    cu_val, data_index_cu = transform_indices(cold_users_history, 'user_id', 'item_id')
    cu_holdout = reindex_data(holdout, data_index_cu, fields="items")

    data_description = dict(
        users=data_index_train['users'].name,
        items=data_index_train['items'].name,
        feedback='watched_pct',
        n_users_train=len(data_index_train['users']),
        n_items=data_index_train['items'].shape[0],  # interactions_df.item_id.nunique(),
        user_features=csr_matrix(
            users_df_ohe[users_df_ohe.user_id.isin(data_index_train['users'])].drop(columns='user_id').values),
        item_features=csr_matrix(
            items_df_ohe[items_df_ohe.item_id.isin(data_index_train['items'])].drop(columns='item_id').values),
        holdout_standard=holdout_val,
        holdout_cs=cu_holdout,
        cold_start_test=cu_val,
    )

    # get interactions matrix
    train_matrix, iid_to_item_id, item_id_to_iid, uid_to_user_id, user_id_to_uid = \
        get_interaction_matrix(train_val, n_items=data_description['n_items'])

    train_matrix_indices = dict(
        iid_to_itemid=iid_to_item_id,
        itemid_to_iid=item_id_to_iid,
        uid_to_user_id=uid_to_user_id,
        user_id_to_uid=user_id_to_uid)

    # cold users
    cold_users_matrix, iid_to_item_id_cu, item_id_to_iid_cu, uid_to_user_id_cu, user_id_to_uid_cu = \
        get_interaction_matrix(cold_users_history, n_items=data_description['n_items'])

    cold_start_matrix_indices = dict(
        iid_to_itemid=iid_to_item_id_cu,
        itemid_to_iid=item_id_to_iid_cu,
        uid_to_user_id=uid_to_user_id_cu,
        user_id_to_uid=user_id_to_uid_cu)

    return train_val, data_description, train_matrix, train_matrix_indices, cold_users_matrix, cold_start_matrix_indices


def leave_last_out(data, userid='userid', timeid='timestamp'):
    data_sorted = data.sort_values(timeid)
    holdout = data_sorted.drop_duplicates(
        subset=[userid], keep='last'
    ) # split the last item from each user's history
    remaining = data.drop(holdout.index) # store the remaining data - will be our training
    return remaining, holdout


def transform_indices(data, users, items):
    '''
    Reindex columns that correspond to users and items.
    New index is contiguous starting from 0.
    '''
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        new_index, data_index[entity] = to_numeric_id(data, field)
        data = data.assign(**{f'{field}': new_index}) # makes a copy of dataset!
    return data, data_index


def to_numeric_id(data, field):
    '''
    Get new contiguous index by converting the data field
    into categorical values.
    '''
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


def reindex_data(data, data_index, fields=None):
    '''
    Reindex provided data with the specified index mapping.
    By default, will take the name of the fields to reindex from `data_index`.
    It is also possible to specify which field to reindex by providing `fields`.
    '''
    if fields is None:
        fields = data_index.keys()
    if isinstance(fields, str): # handle single field provided as a string
        fields = [fields]
    for field in fields:
        entity_name = data_index[field].name
        new_index = data_index[field].get_indexer(data[entity_name])
        data = data.assign(**{f'{entity_name}': new_index}) # makes a copy of dataset!
    return data


# generate training matrix
def generate_interactions_matrix(data, data_description, rebase_users=False):
    '''
    Converts a pandas dataframe with user-item interactions into a sparse matrix representation.
    Allows reindexing user ids, which help ensure data consistency at the scoring stage
    (assumes user ids are sorted in the scoring array).

    Args:
        data (pandas.DataFrame): The input dataframe containing the user-item interactions.
        data_description (dict): A dictionary containing the data description with the following keys:
            - 'n_users' (int): The total number of unique users in the data.
            - 'n_items' (int): The total number of unique items in the data.
            - 'users' (str): The name of the column in the dataframe containing the user ids.
            - 'items' (str): The name of the column in the dataframe containing the item ids.
            - 'feedback' (str): The name of the column in the dataframe containing the user-item interaction feedback.
        rebase_users (bool, optional): Whether to reindex the user ids to make contiguous index starting from 0. Defaults to False.

    Returns:
        scipy.sparse.csr_matrix: A sparse matrix of shape (n_users, n_items) containing the user-item interactions.
    '''

    n_users = data_description['n_users']
    n_items = data_description['n_items']
    # get indices of observed data
    user_idx = data[data_description['users']].values
    if rebase_users:  # handle non-contiguous index of test users
        # This ensures that all user ids are contiguous and start from 0,
        # which helps ensure data consistency at the scoring stage.
        user_idx, user_index = pd.factorize(user_idx, sort=True)
        n_users = len(user_index)
    item_idx = data[data_description['items']].values
    feedback = data[data_description['feedback']].values
    # construct rating matrix
    return csr_matrix((feedback, (user_idx, item_idx)), shape=(n_users, n_items))



def warm_start_timepoint_split(data, time_split_q=0.95):
    """
    Split data into training, testset, and holdout datasets based on a timepoint split
    and according to the `warm-start` evaluation strategy.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset containing columns `userid`, `movieid`, and `timestamp`.
    time_split_q : float, optional
        The quantile value used to split the dataset based on the `timestamp` column.
        Default is 0.95.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple of three pandas DataFrames: training, testset, and holdout.
        `training` is a subset of `data` used for training the recommender system.
        `testset` is a subset of `data` used for generating recommendations for the test users.
        `holdout` is a subset excluded from `testset` containing only the most recent interactions for each test user.

    Notes
    -----
    The function splits the input `data` into three subsets: `training`, `testset`, and `holdout`.
    The split is performed based on the `timestamp` column of `data`, using `time_split_q` as the quantile value.
    The `holdout` dataset contains only the immediate interactions following the fixed timepoint for each test user from the `testset`.
    The set of users in `training` is disjoint with the set of users in the `testset`, which implements the `warm-start` scenario.
    """
    timepoint = data.timestamp.quantile(q=time_split_q, interpolation='nearest')
    test_ = data.query('timestamp >= @timepoint')
    rest_ = data.drop(test_.index)
    holdout_ = (
        test_
        .sort_values('timestamp')
        .drop_duplicates(subset=['userid'], keep='first')
    )
    # the holdout dataframe contains interactions closest to certain timepoint from the right,
    # i.e., the corresponding items are the first in each test user profile after this timepoint
    training = rest_.query('userid not in @holdout_.userid')
    train_items = training.movieid.unique()
    testset_ = rest_.query('userid in @holdout_.userid and movieid in @train_items')
    test_users = testset_.userid.unique()
    holdout = holdout_.query(
        # if user is not in `test_users` then no evluation is possible,
        # if item is not in `train_items` it's cold start -> must be excluded
        'userid in @test_users and movieid in @train_items'
    ).sort_values('userid')
    testset = testset_.query(
        # make sure testset and holdout contain the same set of users
        'userid in @holdout.userid'
    ).sort_values('userid')
    return training, testset, holdout


def get_interaction_matrix(data, n_items, userid='user_id', itemid='item_id', rating='watched_pct'):
    data['uid'] = data[userid].astype('category')
    data['uid'] = data['uid'].cat.codes

    data['iid'] = data[itemid].astype('category')
    data['iid'] = data['iid'].cat.codes
    interactions_vec = csr_matrix((data[rating],
                                   (data['uid'], data['iid'])),
                                  shape=(data.uid.nunique(), n_items))
    # res = interactions_vec.sum(axis=1)
    #     val = np.repeat(res, interactions_vec.getnnz(axis=1))
    #     interactions_vec.data /= np.ravel(val)

    iid_to_item_id = data[['iid', itemid]].drop_duplicates().set_index('iid').to_dict()[itemid]
    item_id_to_iid = data[['iid', itemid]].drop_duplicates().set_index(itemid).to_dict()['iid']

    uid_to_user_id = data[['uid', userid]].drop_duplicates().set_index('uid').to_dict()[userid]
    user_id_to_uid = data[['uid', userid]].drop_duplicates().set_index(userid).to_dict()['uid']
    return interactions_vec, iid_to_item_id, item_id_to_iid, uid_to_user_id, user_id_to_uid
