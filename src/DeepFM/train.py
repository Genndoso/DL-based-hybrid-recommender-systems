from tqdm import tqdm
import numpy as np
import torch 

def epoch_train(loader, model, criterion, opt, device):
    model.train()
    total_loss = 0

    for input, target in loader:
        opt.zero_grad()
        input, target = input.to(device), target.to(device) 
        output = model(input)     

        loss = criterion(output, target.float())
        total_loss +=loss
        loss.backward()
        opt.step()

    return total_loss / len(loader.dataset)


def epoch_test(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    avg_acc = 0

    for input, target in loader:
        input, target = input.to(device), target.to(device) 
        output = model(input)

        loss = criterion(output, target.float())
        print('MODEL OUTPUT:', output)
        print('TARGET:', target.float())
        print(loss)
        print()
        print()
        total_loss += loss
        break

    return total_loss / len(loader.dataset)#, avg_acc / len(loader.dataset)


def evaluate(recommended_items, holdout, holdout_description, topn=20):
    itemid = holdout_description['items']
    holdout_items = holdout[itemid].values
    assert recommended_items.shape[0] == len(holdout_items)
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)

    # MRR calculation
    n_test_users = recommended_items.shape[0]
    hit_rank = np.where(hits_mask)[1] + 1.0
    
    # NDCG
    ndcg_pu = 1.0 / np.log2(hit_rank + 1)
    ndcg = np.sum(ndcg_pu) / n_test_users

    return ndcg


def user_evaluate(user_id, preds, holdout, n_items):
    preds = preds.cpu().detach().numpy()

    holdout_user = holdout[holdout.user_id ==user_id]
    predictions = preds[user_id]
    
    hits_mask = predictions == holdout_user.item_id.values

    # HR calculation
    hr = np.mean(hits_mask.any(axis=0))
    # MRR calculation
    n_test_users = preds.shape[0]
    hit_rank = np.where(hits_mask)[0] + 1.0
    mrr = np.sum(1 / hit_rank) 
    # coverage calculation

    cov = np.unique(preds).size / n_items
    return hr, mrr, cov

def get_unseen_movies(data_df, user_id):
    user_rating = data_df.loc[user_id,:]
    already_seen = user_rating[ user_rating > 0].index.tolist()
    movies_list = data_df.columns.tolist()
    
    unseen_list = [ movie for movie in movies_list if movie not in already_seen]
    return unseen_list

def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    res_dict = {pred_df[i]: i for i in range(len(pred_df))}
    pred = pred_df.sort()
    res_movie_ids = [items[i] for i in result.sort().indices.tolist()]
    recomm_ids = list(set(res_movie_ids) & set(unseen_list))

    return res_movie_ids[:top_n]


def train(device, train_loader, test_loader, model, criterion, opt, n_epochs=10):
    curr_best_loss = None
    for epoch in tqdm(range(n_epochs)):
        train_loss = epoch_train(train_loader, model, criterion, opt, device)
        test_loss = epoch_test(test_loader, model, criterion, device)

        if curr_best_loss is None or test_loss < curr_best_loss:
            curr_best_loss = test_loss
            torch.save(model, "./train_data/dfm.pt")

        print(f'[Epoch {epoch + 1}] train loss: {train_loss:.4f}; ' + 
              f'test loss: {test_loss:.4f}')
        break

    