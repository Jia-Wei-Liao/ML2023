import os
import argparse
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from src.model import DNN
from src.dataset import get_data_loader
from src.trainer import Trainer
from src.constant import FEATURES
from src.utils import set_random_seed

from sklearn.preprocessing import QuantileTransformer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--seed', type=int, default=5201314)
    parser.add_argument('--data_path', type=str, default='dataset/covid_train.csv')
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('-ep', '--n_epoch', type=int, default=3000)
    parser.add_argument('-bs', '--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device_id', type=int, default=0)

    return parser.parse_args()


def train(args):
    train_data = pd.read_csv(args.data_path).values
    X, y = train_data[:, :-1], train_data[:, -1]

    qt = QuantileTransformer(n_quantiles=5)
    X = qt.fit_transform(X)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.valid_ratio, random_state=args.seed)
    train_loader = get_data_loader(X_train, y_train, FEATURES, args.batch_size, args.num_workers, train=True)
    valid_loader = get_data_loader(X_valid, y_valid, FEATURES, args.batch_size, args.num_workers)

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    model = DNN([len(FEATURES), 64, 32, 16, 8, 4]).to(device)
    print(model)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    ## TODO: lr_scheduler
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.3333333333333333, end_factor=1.0, total_iters=5, last_epoch=- 1, verbose=False)

    trainer = Trainer(
        train_loader,
        valid_loader,
        model,
        criterion,
        optimizer,
        args.n_epoch,
        device
    )
    trainer.fit()

    return X_train, X_valid, y_train, y_valid


if __name__ == "__main__":
    args = parse_arguments()
    set_random_seed(args.seed)
    train(args)
