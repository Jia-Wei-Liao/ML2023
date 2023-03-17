import os
import argparse
import pandas as pd

import torch
import torch.nn as nn

from src.model import DNN
from src.dataset import get_data_loader
from src.constant import FEATURES
from src.utils import set_random_seed, save_csv

from sklearn.preprocessing import QuantileTransformer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--seed', type=int, default=5201314)
    parser.add_argument('--data_path', type=str, default='dataset/covid_test.csv')
    parser.add_argument('-bs', '--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)

    return parser.parse_args()


@torch.no_grad()
def infer(args):
    X_test = pd.read_csv(args.data_path).values
    test_loader = get_data_loader(X_test, features=FEATURES, batch_size=args.batch_size,
                                  num_workers=args.num_workers, train=False)
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    model = DNN([len(FEATURES), 16, 8]).to(device)
    checkpoint = torch.load('checkpoint.ckpt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    preds = []
    for id_, data in enumerate(test_loader):
        pred = model(data['X'].to(device))
        preds.extend(list(pred.detach().cpu().numpy()))

    save_csv([{'id': id_, 'tested_positive': pred} for id_, pred in enumerate(preds)], 'submissions.csv')
    return


if __name__ == "__main__":
    args = parse_arguments()
    set_random_seed(args.seed)
    infer(args)
