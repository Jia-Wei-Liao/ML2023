import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict


class COVID19Dataset(Dataset):
    def __init__(self, X, y=None, X_mean=None, X_std=None):
        self.X = torch.FloatTensor(X)
        self.y = y if y is None else torch.FloatTensor(y)

    def select_features(self, features):
        self.X = self.X[:, features]
        return

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self. y is not None:
            return OrderedDict(X=self.X[idx], y=self.y[idx], log_y=torch.log(self.y[idx]))
        else:
            return OrderedDict(X=self.X[idx])


def get_data_loader(X, y=None, features=[], batch_size=1, num_workers=1, train=False):
    dataset = COVID19Dataset(X, y)
    if features:
        dataset.select_features(features)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                   shuffle=True if train else False, pin_memory=True)
    return data_loader
