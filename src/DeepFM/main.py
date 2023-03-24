import csv
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from random import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

from data_prep import (
    prepare_loader,
    prepare_topn_loader,
    generate_embeddings,
    get_input_dims,
    timepoint_split,
)
from train import get_seen_items, recomm_movie_by_userid, user_evaluate
from train import train
from deepfm import DeepFM
from comet_ml import Experiment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train",
    default=True,
    type=bool
)

parser.add_argument(
    "--cold_start",
    default=False,
    type=bool
)


def set_up_exp():
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="Dcj59uwP496tLKvrjrWEU79R0",
        project_name="recsys-deepfm",
        workspace="highly0",
    )
    return experiment


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_train = args.train
    target = "watched_pct"

    if is_train: # training from scratch
        dataset_path = "../../dataset/data.csv"
        data_df = pd.read_csv(dataset_path)
        data_df.drop(['item_title', 'item_description', 'user_kids_flg'], axis=1, inplace=True)

        like_criteria = 50
        data_df.loc[data_df[target] < like_criteria, target] = 0
        data_df.loc[data_df[target] >= like_criteria, target] = 1

        # getting our embeddings
        data_df, feature_embeddings = generate_embeddings(data_df, target)

        # getting our train, test and holdout and prepare our dataloaders
        train_df, test_df, holdout_df = timepoint_split(data_df, time_split_q=0.95)
        train_df.to_csv("./train_data/TRAIN.csv", index=False)
        test_df.to_csv("./train_data/TEST.csv", index=False)
        holdout_df.to_csv("./train_data/HOLDOUT.csv", index=False)

        # Create an experiment with your api key
        experiment = set_up_exp()

        train_loader, test_loader = prepare_loader(
            train_df, test_df, feature_embeddings, target
        )
        input_dims = get_input_dims(feature_embeddings)
        embedding_dim = 10
        model = DeepFM(input_dims, embedding_dim, mlp_dims=[30, 20, 10]).to(device)
        bce_loss = nn.BCELoss()  # Binary Cross Entropy loss
        lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train(
            experiment,
            device,
            train_loader,
            test_loader,
            model,
            bce_loss,
            optimizer,
            n_epochs=10,
        )
    else: # validating
        model = torch.load("./train_data/dfm.pt")
        model.eval()

        cold_start = args.cold_start
        holdout_df = pd.read_csv("./train_data/HOLDOUT.csv")
        holdout_df = holdout_df[holdout_df[target] == 1]
        test_df = pd.read_csv("./train_data/TEST.csv")
        if cold_start: # testing cold_start
            num_interaction_pu = test_df.groupby('user_id')['item_id'].count().sort_values(ascending=False)
            # get cold_users
            users = list(num_interaction_pu.loc[(num_interaction_pu < 5) & (num_interaction_pu > 2)].index)
            shuffle(users)
        else: # warm start
            users = list(holdout_df["user_id"].unique())
            shuffle(users)

        scores = {'hr': [], 'mrr': [], 'cov': []}

        for u in tqdm(users):
            u = int(u)
            try:
                loader = prepare_topn_loader(u, test_df, target)
                result = torch.tensor([]).to(device)
                for input, _ in loader:
                    input = input.to(device)
                    output = model(input)
                    result = torch.cat((result, output), 0)
                result_list = result.tolist()

                seen_list = get_seen_items(test_df, u)
                rec = recomm_movie_by_userid(result_list, seen_list, top_n=20)

                n_items = len(test_df['item_id'].unique())
                hr, mrr, cov = user_evaluate(u, rec, holdout_df, n_items)

                scores['hr'].append(hr)
                scores['mrr'].append(mrr)
                scores['cov'].append(cov)
            except KeyboardInterrupt:
                print('exiting early from evaluating')
                break

        hr_mean = sum(scores['hr']) / len(scores['hr'])
        mrr_mean = sum(scores['mrr']) / len(scores['mrr'])
        cov_mean = sum(scores['cov']) / len(scores['cov'])

        print('HR:', hr_mean)
        print('MRR:', mrr_mean)
        print('COV:', cov_mean)
