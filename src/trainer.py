import numpy as np

import torch
from tqdm import tqdm

from src.utils import AverageMeter, Logger


class Trainer():
    def __init__(
        self,
        train_loader,
        valid_loader,
        model,
        criterion,
        optimizer,
        n_epoch,
        device):

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epoch = n_epoch
        self.device = device
        self.train_record = AverageMeter()
        self.valid_record = AverageMeter()
        self.log = Logger('record.csv')

    def share_step(self, X, y, log_y):
        X, y, log_y = X.to(self.device), y.to(self.device), log_y.to(self.device)
        pred = self.model(X)
        loss = self.criterion(pred, y)
        mse = self.criterion(torch.exp(pred), y)
        return loss, mse

    def train_step(self):
        self.train_record.reset()
        self.model.train()
        for data in self.train_loader:
            X, y, log_y = data['X'], data['y'], data['log_y']
            self.optimizer.zero_grad()
            loss, mse = self.share_step(X, y, log_y)
            loss.backward()
            self.optimizer.step()
            self.train_record.update(loss.detach().item(), X.shape[0])
        return

    @torch.no_grad()
    def valid_step(self):
        self.valid_record.reset()
        self.model.eval()
        for data in self.valid_loader:
            X, y, log_y = data['X'], data['y'], data['log_y']
            loss, mse = self.share_step(X, y, log_y)
            self.valid_record.update(loss.detach().item(), X.shape[0])
        return

    def fit(self):
        best_loss = np.Inf
        progress_bar = tqdm(range(1, self.n_epoch+1))
        for self.cur_ep in progress_bar:
            self.train_step()
            self.valid_step()
            progress_bar.set_postfix(
                train_loss=self.train_record.avg,
                valid_loss=self.valid_record.avg
            )
            self.log.add(
                train_loss=self.train_record.avg,
                valid_loss=self.valid_record.avg
            )
            if self.valid_record.avg < best_loss:
                torch.save(self.model.state_dict(), 'checkpoint.ckpt')

        self.log.save()
        return
