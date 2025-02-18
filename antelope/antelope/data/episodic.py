from dataclasses import dataclass

from absl import flags

import numpy as np
import torch

FLAGS = flags.FLAGS


def load_data_flags():
    data = np.load(FLAGS.data_file, allow_pickle=True)
    return data["flags"].item()


def batch_iterator(X, y, batch_size):
    i = 0
    n = X.shape[0]
    while i < n:
        yield X[i : i + batch_size], y[i : i + batch_size]
        i += batch_size


def batch_stream(X, y, batch_size):
    n = X.shape[0]
    while True:
        perm = torch.randperm(n)
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        yield from batch_iterator(X_shuffled, y_shuffled, batch_size)


@dataclass(frozen=True)
class EpisodicSplit:
    X: torch.Tensor
    y: torch.Tensor

    def iterator(self, batch_size: int, fixed=False):
        if fixed:
            return batch_iterator(self.X, self.y, batch_size)
        else:
            return batch_stream(self.X, self.y, batch_size)


class EpisodicDataset:
    def __init__(self, data_file: str):
        data = np.load(data_file, allow_pickle=True)
        self.train = EpisodicSplit(
            torch.from_numpy(data["train_X"]), torch.from_numpy(data["train_z"]).long()
        )
        self.test = EpisodicSplit(
            torch.from_numpy(data["test_X"]), torch.from_numpy(data["test_z"]).long()
        )

    @classmethod
    def from_flags(cls):
        return cls(FLAGS.data_file)

    @property
    def in_dim(self):
        return self.train.X.size(-1)

    @property
    def out_dim(self):
        return self.train.y.size(-1)
