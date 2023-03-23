import torch



class TestDataset(torch.utils.data.Dataset):
    def __init__(self, userInput, itemInput):
        super(TestDataset, self).__init__()
        self.userInput = torch.Tensor(userInput).long()
        self.itemInput = torch.Tensor(itemInput).long()

    def __getitem__(self, index):
        return self.userInput[index], self.itemInput[index]

    def __len__(self):
        return self.userInput.size(0)


def get_optimizer(name, lr, scope):
    if name.lower() == "adagrad":
        return torch.optim.Adagrad(scope, lr=lr)
    elif name.lower() == "rmsprop":
        return torch.optim.RMSprop(scope, lr=lr)
    elif name.lower() == "adam":
        return torch.optim.Adam(scope, lr=lr)
    elif name.lower() == "sgd":
        return torch.optim.SGD(scope, lr=lr)
    else:
        raise ValueError(f"{name} optimizer is not supported!")