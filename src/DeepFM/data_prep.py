from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset


class RatingDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input_tensor = input_tensor.long()
        self.target_tensor = target_tensor.long()

    def __getitem__(self, index):
        return (
            self.input_tensor[index],
            self.target_tensor[index],
        )

    def __len__(self):
        return self.target_tensor.size(0)


def generate_embedding(obj, data_df):
    """
    given object, create a mapping from 0->n where n is the number
    of distinct types in object. This will be later used to create
    our embedding to feed into the net
    """
    number_objects = list(set(data_df.loc[:, obj]))
    number_objects.sort()

    # map from each object to its distinct 0->n id
    object_dict = {number_objects[i]: i for i in range(len(number_objects))}
    data_df[obj] = data_df[obj].map(lambda x: object_dict[x])

    return data_df, object_dict, len(number_objects)


def generate_embeddings(data_df, target):
    """
    given a list of features, use "generate_embedding" for all features
    """
    print("generating embeddings!")
    features = list(data_df.columns)
    features.remove(target)
    feature_embeddings = {}

    for feature in tqdm(features):
        data_df, curr_dict, curr_len = generate_embedding(feature, data_df)
        feature_embeddings[feature] = {}
        feature_embeddings[feature]["mapping_dict"] = curr_dict
        feature_embeddings[feature]["length"] = curr_len
    print("finished generating embeddings")
    return data_df, feature_embeddings


def get_input_dims(feature_embeddings):
    """
    return the length of all of our mapping from
    feature embeddings
    """
    embeddings = []
    for _, val in feature_embeddings.items():
        embeddings.append(val["length"])
    return embeddings


def timepoint_split(data, time_split_q=0.95):
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
    timepoint = data.timestamp.quantile(q=time_split_q, interpolation="nearest")
    test_ = data.query("timestamp >= @timepoint")
    rest_ = data.drop(test_.index)
    holdout_ = test_.sort_values("timestamp").drop_duplicates(
        subset=["user_id"], keep="first"
    )
    # the holdout dataframe contains interactions closest to certain timepoint from the right,
    # i.e., the corresponding items are the first in each test user profile after this timepoint
    training = rest_.query("user_id not in @holdout_.user_id")
    train_items = training.item_id.unique()
    testset_ = rest_.query("user_id in @holdout_.user_id and item_id in @train_items")
    test_users = testset_.user_id.unique()
    holdout = holdout_.query(
        # if user is not in `test_users` then no evluation is possible,
        # if item is not in `train_items` it's cold start -> must be excluded
        "user_id in @test_users and item_id in @train_items"
    ).sort_values("user_id")
    testset = testset_.query(
        # make sure testset and holdout contain the same set of users
        "user_id in @holdout.user_id"
    ).sort_values("user_id")
    return training, testset, holdout


def torchify(features, df, target):
    """
    turn the features of provided df into a torch tensor.
    return x and y (tensors) ready for dataloader prepping
    """
    cols = []

    for feature in tqdm(features):
        feature_col = torch.tensor(df.loc[:, feature].values)
        feature_col = feature_col.unsqueeze(1)
        cols.append(feature_col)
    X = torch.cat(cols, dim=1)
    y = torch.tensor(list(df.loc[:, target]))

    return X, y


def torchify_topn(features, user_features_df, item_features_df):
    user_features = list(user_features_df.columns)
    item_features = list(item_features_df.columns)
    length = len(item_features_df)
    cols = []
    for feature in features:
        if feature in user_features:
            # duplicate user for all items
            feature_col = torch.tensor([user_features_df[feature].values[0]] * length)
        elif feature in item_features:
            feature_col = torch.tensor(item_features_df.loc[:, feature].values)
        feature_col = feature_col.unsqueeze(1)
        cols.append(feature_col)

    X = torch.cat(cols, dim=1)
    y = torch.tensor([0] * length)
    return X, y


def prepare_topn_loader(user_id, data_df, target):
    user_info = data_df[data_df["user_id"] == user_id].head(1)
    user_info = user_info.drop(data_df.columns[7:], axis=1)
    user_info.drop([target], axis=1, inplace=True)

    item_info = data_df.copy()
    item_info = item_info.drop(item_info.columns[:7], axis=1)

    item_info = data_df.drop(data_df.columns[:7], axis=1)
    item_info = item_info.drop_duplicates(keep="first")

    features = list(data_df.columns)
    features.remove(target)
    X, y = torchify_topn(features, user_info, item_info)
    dataset = RatingDataset(X, y)
    loader = DataLoader(dataset, batch_size=2024, shuffle=False)
    return loader


def prepare_loader(train_df, test_df, feature_embeddings, target):
    """
    process our data and return train and validate dataloaders
    """
    # normal training, create our train and test laoder
    X_train, y_train = torchify(list(feature_embeddings.keys()), train_df, target)
    train_dataset = RatingDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    X_test, y_test = torchify(list(feature_embeddings.keys()), test_df, target)
    test_dataset = RatingDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    return train_loader, test_loader
