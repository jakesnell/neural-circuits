from datetime import datetime
import os
from collections import namedtuple
from functools import partial
import random

from absl import app
from absl import flags
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from antelope.data import EpisodicDataset, CRPDataset
from antelope.loss import BatchedCrossEntropy
from antelope.optim.adam import adam

from antelope.models.rnn import GRU

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 50, "size of batches for evaluation")
flags.DEFINE_integer("max_iter", 5000, "number of learning iterations")
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_string("device", "cpu", "which device to run on")
flags.DEFINE_bool("compile", False, "whether to compile the model")
flags.DEFINE_bool("plot", False, "whether to plot training curve")
flags.DEFINE_bool("debug", False, "whether to print debugging information")
flags.DEFINE_float("gamma", 0.1, "How much to cut learning rate")
flags.DEFINE_string("out_dir", os.getcwd(), "base path for output")
flags.DEFINE_string("job_id", None, "name of directory within checkpoint dir")
flags.DEFINE_string(
    "data_type",
    "episodic",
    "type of data: episodic => precomputed episodes, crp => sampled via crp",
)


def safe_path(file_name):
    if os.path.exists(file_name):
        i = 1
        base, ext = os.path.splitext(file_name)
        while os.path.exists(base + f".{i:d}" + ext):
            i += 1
        return base + f".{i:d}" + ext
    else:
        return file_name


def safe_out_dir():
    if FLAGS.job_id is None:
        job_id = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    else:
        job_id = FLAGS.job_id

    res = os.path.join(FLAGS.out_dir, job_id)
    os.mkdir(res)
    return res


StepResult = namedtuple("StepResult", "loss logits")


def train_step(data_batch, model, criterion, optimizer):
    X, y = data_batch
    X = X.to(device=torch.device(FLAGS.device))
    y = y.to(device=torch.device(FLAGS.device))
    optimizer.zero_grad()
    logits = model(X, y)
    loss = criterion(logits, y, batch_reduction="mean", time_reduction="mean")
    loss.backward()
    optimizer.step()

    return StepResult(loss=loss.detach().cpu(), logits=logits.detach().cpu())


def main(_):
    batch_size = FLAGS.batch_size
    max_iter = FLAGS.max_iter
    torch_compile = FLAGS.compile

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    if FLAGS.data_type == "episodic":
        dataset = EpisodicDataset.from_flags()
    elif FLAGS.data_type == "crp":
        dataset = CRPDataset.from_flags()
    else:
        raise ValueError(f"unknown data type {FLAGS.data_type}")

    model = GRU.from_flags(dataset.in_dim, dataset.out_dim)
    model.to(torch.device(FLAGS.device))
    if torch_compile:
        model = torch.compile(model)

    criterion = BatchedCrossEntropy()
    optimizer = adam(model.parameters())

    step = partial(train_step, model=model, criterion=criterion, optimizer=optimizer)

    scheduler = StepLR(optimizer, step_size=1000, gamma=FLAGS.gamma)

    loss_vals = []
    for _, data_batch in zip(
        pbar := tqdm(range(1, max_iter + 1)), dataset.train.iterator(batch_size)
    ):
        result = step(data_batch)
        pbar.set_description(
            f"loss={result.loss:0.6f}, perplexity={torch.exp(result.loss):0.6f}, lr={scheduler.get_last_lr()[0]:0.1e}"
        )
        loss_vals.append(result.loss)
        scheduler.step()

    loss_vals = torch.tensor(loss_vals)

    print(f"avg final losses: {loss_vals[-100:].mean():0.6f}")

    if FLAGS.plot:
        plt.plot(loss_vals)
        plt.show()

    if FLAGS.debug:
        counts = torch.bincount(dataset.train.y.reshape(-1))
        counts = counts / counts.sum()

        df = pd.merge(
            pd.DataFrame({"class": torch.arange(counts.size(-1)), "empirical": counts}),
            pd.DataFrame(
                {
                    "class": torch.arange(dataset.out_dim),
                    "predicted": F.softmax(
                        result.logits.detach(), -1  # pyright: ignore
                    )
                    .view(-1, result.logits.size(-1))  # pyright: ignore
                    .mean(0),
                }
            ),
            on="class",
        )
        df[["empirical", "predicted"]].plot(kind="bar")
        plt.xlabel("class")
        plt.show()

    out_dir = safe_out_dir()

    torch.save(loss_vals, os.path.join(out_dir, "loss_vals.pt"))
    torch.save(model.state_dict(), os.path.join(out_dir, "circuit_model.pt"))
    torch.save(FLAGS.flag_values_dict(), os.path.join(out_dir, "flags.pt"))


if __name__ == "__main__":
    app.run(main)
