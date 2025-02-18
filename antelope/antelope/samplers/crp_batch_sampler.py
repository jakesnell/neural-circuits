from typing import NamedTuple, List

import numpy as np

import torch
from torch.utils.data import SubsetRandomSampler, default_collate

from numba import njit


@njit
def crp_sample(alpha: float, num_reps: int, num_timesteps: int):
    inds = np.empty((num_reps, num_timesteps), dtype=np.int64)

    for i in range(num_reps):
        counts = np.zeros(num_timesteps + 1)
        counts[0] = alpha

        k = 0
        u = np.random.rand(num_timesteps) * (np.arange(num_timesteps) + alpha)

        for n in range(num_timesteps):
            for j in range(num_timesteps):
                if u[n] > counts[j]:
                    u[n] -= counts[j]
                else:
                    break

            counts[j] += 1.0
            if j == k:
                counts[j] = 1.0
                counts[k + 1] = alpha
                k += 1

            inds[i, n] = j

    return inds


class CRPBatchSampler(NamedTuple):
    class_datasets: List
    alpha: float
    batch_size: int
    sequence_length: int

    def decode(self, trajectory: List[int]):
        class_iter = iter(SubsetRandomSampler(self.class_datasets))
        data_iters = []
        ret = []

        for class_ind in trajectory:
            if class_ind >= len(data_iters):
                data_iters.append(iter(SubsetRandomSampler(next(class_iter))))

            ret.append(next(data_iters[class_ind]))
        return default_collate(ret)

    def sample(self):
        trajectories = torch.from_numpy(
            crp_sample(self.alpha, self.batch_size, self.sequence_length)
        )
        return (
            default_collate(
                [self.decode(trajectory) for trajectory in trajectories.tolist()]
            ),
            trajectories,
        )

    def __iter__(self):
        while True:
            yield self.sample()
