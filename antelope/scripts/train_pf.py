from datetime import datetime
import os
from collections import namedtuple
from functools import partial
import random

from absl import app
from absl import flags
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np

import matplotlib.pyplot as plt


from antelope.data import EpisodicDataset, CRPDataset
from antelope.models.exp_family import (
    ExponentialFamilyParticleFilter,
    FourParameterExponentialFamilyParticleFilter,
)
from antelope.optim.adam import adam

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 50, "size of batches for evaluation")
flags.DEFINE_integer("max_iter", 5000, "number of learning iterations")
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_bool("plot", False, "whether to plot training curve")
flags.DEFINE_bool("debug", False, "whether to print debugging information")
flags.DEFINE_bool("resume", False, "whether to resume from a previous checkpoint")
flags.DEFINE_bool(
    "four_param", True, "whether to use four parameter normal inverse gamma"
)
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
    if not os.path.exists(res):
        os.mkdir(res)
    return res


StepResult = namedtuple("StepResult", "loss")


def train_step(data_batch, model, optimizer, cuda: bool):
    X, y = data_batch
    if cuda:
        X = X.cuda()
        y = y.cuda()
    optimizer.zero_grad()
    loss = model.loss(X, y)
    loss.backward()
    optimizer.step()

    return StepResult(loss=loss.detach().cpu())


def main(_):
    cuda = torch.cuda.is_available()
    batch_size = FLAGS.batch_size
    max_iter = FLAGS.max_iter
    four_param = FLAGS.four_param
    resume = FLAGS.resume

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    if FLAGS.data_type == "episodic":
        dataset = EpisodicDataset.from_flags()
    elif FLAGS.data_type == "crp":
        dataset = CRPDataset.from_flags()
    else:
        raise ValueError(f"unknown data type {FLAGS.data_type}")

    if four_param:
        print("creating four parameter model...")
        model = FourParameterExponentialFamilyParticleFilter(
            dataset.in_dim, alpha=FLAGS.alpha
        )
    else:
        print("creating ordinary model...")
        model = ExponentialFamilyParticleFilter(dataset.in_dim, alpha=FLAGS.alpha)
    if cuda:
        model.cuda()
    optimizer = adam(model.parameters())

    out_dir = safe_out_dir()

    if resume:
        print("resuming from checkpoint...")
        model_file = os.path.join(out_dir, "expfamily_model_last.pt")
        model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))

        optimizer_file = os.path.join(out_dir, "optimizer_last.pt")
        optimizer.load_state_dict(
            torch.load(optimizer_file, map_location=torch.device("cpu"))
        )

    step = partial(train_step, model=model, optimizer=optimizer, cuda=cuda)

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

        torch.save(loss_vals, os.path.join(out_dir, "loss_vals.pt"))
        torch.save(model.state_dict(), os.path.join(out_dir, "expfamily_model_last.pt"))
        torch.save(optimizer.state_dict(), os.path.join(out_dir, "optimizer_last.pt"))

    loss_vals = torch.tensor(loss_vals)

    print(f"avg final losses: {loss_vals[-100:].mean():0.6f}")

    if FLAGS.plot:
        plt.plot(loss_vals)
        plt.show()

    torch.save(loss_vals, os.path.join(out_dir, "loss_vals.pt"))
    torch.save(model.state_dict(), os.path.join(out_dir, "expfamily_model.pt"))
    torch.save(FLAGS.flag_values_dict(), os.path.join(out_dir, "flags.pt"))


if __name__ == "__main__":
    app.run(main)
