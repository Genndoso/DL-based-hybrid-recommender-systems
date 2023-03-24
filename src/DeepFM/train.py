from tqdm import tqdm
import numpy as np
import torch


def log_experiment(experiment, epoch, train_loss, test_loss):
    experiment.log_metric("Train Loss", train_loss, step=epoch)
    experiment.log_metric("Val Loss", test_loss, step=epoch)


def epoch_train(loader, model, criterion, opt, device):
    model.train()
    total_loss = 0
    counter = 0

    for input, target in loader:
        opt.zero_grad()
        input, target = input.to(device), target.to(device)
        output = model(input)

        loss = criterion(output.float(), target.float())

        total_loss += loss.item()
        loss.backward()
        opt.step()
        counter += 1

    return total_loss / counter


def epoch_test(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    counter = 0

    for input, target in loader:
        input, target = input.to(device), target.to(device)
        output = model(input)

        loss = criterion(output.float(), target.float())
        total_loss += loss
        counter += 1
    return total_loss / counter


def user_evaluate(user_id, preds, holdout, n_items):
    holdout_user = holdout[holdout['user_id'] == user_id]
    hits_mask = preds == holdout_user.item_id.values

    # HR calculation
    hr = np.mean(hits_mask.any(axis=0))

    # MRR calculation
    #n_test_users = preds.shape[0]
    hit_rank = np.where(hits_mask)[0] + 1.0
    mrr = np.sum(1 / hit_rank)

    # coverage calculation
    cov = np.unique(preds).size / n_items
    return hr, mrr, cov


def get_seen_items(data_df, user_id):
    seen = set(list(data_df[data_df["user_id"] == user_id].item_id))
    return seen


def recomm_movie_by_userid(preds, seen_list, top_n=10):
    # movies that we have not seen
    pred_dict = {}
    for i in range(len(preds)):
        if i not in seen_list:
            pred_dict[preds[i]] = i

    preds.sort(reverse=True)
    
    res = []
    num = 0
    for pred in preds:
        if pred in pred_dict and pred_dict[pred] not in res:
            res.append(pred_dict[pred])
            num += 1
        if num == top_n:
            break
    return res


def train(
    experiment, device, train_loader, test_loader, model, criterion, opt, n_epochs=10
):
    curr_best_loss = None
    for epoch in tqdm(range(n_epochs)):
        train_loss = epoch_train(train_loader, model, criterion, opt, device)
        test_loss = epoch_test(test_loader, model, criterion, device)

        if curr_best_loss is None or test_loss < curr_best_loss:
            curr_best_loss = test_loss
            torch.save(model, "./train_data/dfm.pt")

        print(
            f"[Epoch {epoch + 1}] train loss: {train_loss:.4f}; "
            + f"test loss: {test_loss:.4f}"
        )

        log_experiment(experiment, epoch, train_loss, test_loss)
