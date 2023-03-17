import csv
import torch
import numpy as np


def set_random_seed(seed):
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.count = 0
        self.total = 0
        self.avg = 0
        return

    def update(self, value, batch_size=1):
        self.value = value
        self.count += batch_size
        self.total += value * batch_size
        self.avg = self.total / self.count


class Logger():
    def __init__(self, path):
        self.path = path
        self.records = []

    def add(self, **inputs):
        self.records.append(inputs)
        return

    def save(self):
        save_csv(self.records, self.path)
        return


def save_csv(data_list, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_list[0].keys())
        writer.writeheader()
        writer.writerows(data_list)
    return
