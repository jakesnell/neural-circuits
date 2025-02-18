import itertools
from typing import List, NamedTuple
import numpy as np
import torch

from antelope.samplers import CRPBatchSampler

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("sequence_length", 100, "length of sequences to train on")
flags.DEFINE_float("alpha", 1.0, "crp coefficient")


def get_class_label(s):
    return s.split("/")[1]


def extract_datasets(x: np.ndarray, y: list):
    inds = {}
    for i, class_label in enumerate(y):
        if class_label not in inds:
            inds[class_label] = []
        inds[class_label].append(i)

    class_labels = sorted(list(inds.keys()))

    return [x[torch.tensor(inds[k])] for k in class_labels], class_labels


class CRPSplit(NamedTuple):
    class_datasets: List
    class_labels: List
    sequence_length: int
    alpha: float

    def iterator(self, batch_size, max_iter=None):
        if max_iter is None:
            max_iter = FLAGS.max_iter

        sampler = CRPBatchSampler(
            self.class_datasets, self.alpha, batch_size, self.sequence_length
        )
        return itertools.islice(iter(sampler), max_iter)


class CRPDataset:
    def __init__(self, data_file: str, sequence_length: int, alpha: float):
        self.sequence_length = sequence_length
        self.alpha = alpha

        data = np.load(data_file, allow_pickle=True)
        X = torch.from_numpy(data["feats"])
        labels = list(map(get_class_label, data["filenames"]))

        class_datasets, class_labels = extract_datasets(X, labels)
        num_classes = len(class_datasets)
        assert num_classes == len(class_labels)

        # TODO: configure this, but for now take half for train and half for test
        self.train = CRPSplit(
            class_datasets[: num_classes // 2],
            class_labels[: num_classes // 2],
            self.sequence_length,
            self.alpha,
        )
        self.test = CRPSplit(
            class_datasets[num_classes // 2 :],
            class_labels[num_classes // 2 :],
            self.sequence_length,
            self.alpha,
        )

    @classmethod
    def from_flags(cls):
        return cls(FLAGS.data_file, FLAGS.sequence_length, FLAGS.alpha)

    @property
    def in_dim(self):
        return self.train.class_datasets[0].size(-1)

    @property
    def out_dim(self):
        return self.sequence_length
