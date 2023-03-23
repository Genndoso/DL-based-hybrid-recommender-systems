import csv
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

from .data_prep import prepare_loader, generate_embeddings, \
                        get_input_dims, timepoint_split, torchify, \
                        RatingDataset
from .train import train
from .deepfm import DeepFM

if __name__ == "__main__":
    target = 'watched_pct'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = '../../dataset/data_dt.csv'
    data_df = pd.read_csv(dataset_path)
    scaler = MinMaxScaler()
    data_df[target] = scaler.fit_transform(data_df[[target]]) # scaling our target

    # getting our embeddings 
    data_df, feature_embeddings = generate_embeddings(data_df, target)

     # getting our train, test and holdout and prepare our dataloaders
    train_df, test_df,  holdout_df = timepoint_split(data_df, time_split_q=0.95)
    train_df.to_csv('./train_data/TRAIN.csv', index=False)
    test_df.to_csv('./train_data/TEST.csv', index=False)
    holdout_df.to_csv('./train_data/HOLDOUT.csv', index=False)

    is_train = True
    if is_train: 
        train_loader, test_loader = prepare_loader(train_df, test_df, feature_embeddings, target)
        
        input_dims = get_input_dims(feature_embeddings)
        embedding_dim = 10
        model = DeepFM(input_dims, embedding_dim, mlp_dims=[30, 20, 10]).to(device)
        bce_loss = nn.MSELoss() #nn.BCELoss() # Binary Cross Entropy loss
        lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train(device, train_loader, test_loader, model, bce_loss, optimizer, n_epochs=2)
    else:
        holdout_df = pd.read_csv('./train_data/HOLDOUT.csv')
        users = holdout_df['user_id'].values

        model = torch.load('dfm.pt')
        model.eval()

        for u in tqdm(users):
            curr_user = holdout_df[holdout_df['user_id'] ==  int(u)]

            train_loader, test_loader = prepare_loader(train_df, test_df, feature_embeddings, target)

            result = torch.tensor([]).to(device)
            for input, _ in test_loader:
                input = input.to(device)
                output = model(input)
                result = torch.cat((result, output), 0)
            
            print(result)
            break
                    

