import click
import os
import abc

import time
import math
from functools import partial
import itertools
import tqdm

import torch
import numpy as np

import jax
import jax.numpy as jnp

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from antelope.loss import BatchedCrossEntropy
from antelope.models.rnn import GRU
from antelope.samplers.crp_sampler import CRPSampler
from antelope.models.exp_family import (
    ExponentialFamilyParticleFilter,
    FourParameterExponentialFamilyParticleFilter,
    FourParameterNoHurdleExponentialFamilyParticleFilter,
)

from partikel.markov_process import ChineseRestaurantProcess
from partikel.feynman_kac import GuidedFeynmanKac
from partikel.particle_filter import ParticleFilter
from partikel.resampler import MultinomialResampler, AdaptiveResampler
import partikel.ppl as ppl


class Evaluator(abc.ABC):
    @abc.abstractmethod
    def nll_trace(self, X: np.ndarray, z: np.ndarray) -> np.ndarray:
        """output the nll trace given the input"""

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """make a map prediction given the input"""


def chunked(x, batch_size):
    n = x.shape[0]

    i = 0
    while i < n:
        yield x[i : i + batch_size]
        i += batch_size


def n_chunks(x, batch_size):
    return int(math.ceil(x.shape[0] / batch_size))


def chunked_iterator(X, z, batch_size):
    return tqdm.tqdm(
        zip(*map(partial(chunked, batch_size=batch_size), [X, z])),
        total=n_chunks(X, batch_size),
    )


def extract_nll_trace(
    evaluator: Evaluator, X: torch.Tensor, z: torch.Tensor, batch_size: int
):
    nll_traces = []

    for X_batch, z_batch in chunked_iterator(X, z, batch_size):
        nll_traces.append(evaluator.nll_trace(X_batch, z_batch))

    return np.concatenate(nll_traces, 0)


def extract_clustering_results(
    evaluator: Evaluator, X: torch.Tensor, z: torch.Tensor, batch_size: int
):
    map_trajectories = []
    rand_scores = []
    mutual_info_scores = []
    map_predict_time = 0.0

    for X_batch, z_batch in chunked_iterator(X, z, batch_size):
        with torch.no_grad():
            batch_start = time.time()
            z_hat = evaluator.predict(X_batch)
            map_predict_time += time.time() - batch_start

            map_trajectories.append(z_hat)

            rand_scores.append(
                np.array(
                    list(itertools.starmap(adjusted_rand_score, zip(z_batch, z_hat)))
                )
            )

            mutual_info_scores.append(
                np.array(
                    list(
                        itertools.starmap(
                            adjusted_mutual_info_score, zip(z_batch, z_hat)
                        )
                    )
                )
            )

    map_trajectories = np.stack(map_trajectories, 0)
    rand_scores = np.concatenate(rand_scores, 0)
    mutual_info_scores = np.concatenate(mutual_info_scores, 0)

    return {
        "map_trajectory": map_trajectories,
        "adjusted_rand": rand_scores,
        "adjusted_mutual_info": mutual_info_scores,
        "map_predict_time": map_predict_time,
        "num_examples": X.shape[0],
    }


class GRUEvaluator(Evaluator):
    def __init__(self, model_file: str, in_dim: int):
        if torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False

        self.model = GRU(in_dim=in_dim, out_dim=100, hidden_size=1024, num_layers=2)
        self.model.load_state_dict(
            torch.load(model_file, map_location=torch.device("cpu"))
        )
        self.model.eval()
        if self.cuda:
            self.model.cuda()
        self.criterion = BatchedCrossEntropy()

    def nll_trace(self, X: np.ndarray, z: np.ndarray) -> np.ndarray:
        """output the nll_trace given the input"""
        X_torch = torch.from_numpy(X)
        z_torch = torch.from_numpy(z).long()
        if self.cuda:
            X_torch = X_torch.cuda()
            z_torch = z_torch.cuda()

        with torch.no_grad():
            logits = self.model(X_torch, z_torch)
            ret = self.criterion(
                logits,
                z_torch,
                batch_reduction="none",
                time_reduction="none",
            )

        return ret.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """make a map prediction given the input"""
        X = torch.from_numpy(X)
        if self.cuda:
            X = X.cuda()
        with torch.no_grad():
            pred = self.model.predict(X)

        return pred.cpu().numpy()


class CRPEvaluator(Evaluator):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.crp_sampler = CRPSampler(alpha, safe_mode=False)
        self.compute_nll = jax.jit(
            jax.vmap(
                lambda z: -self.crp_sampler.log_likelihood(z, time_reduction="none")
            )
        )

        self.partikel_crp = ChineseRestaurantProcess(alpha)

    def nll_trace(self, X: np.ndarray, z: np.ndarray) -> np.ndarray:
        z_jax = jnp.asarray(z)
        return np.asarray(self.compute_nll(z_jax))

    def predict(self, X: np.ndarray) -> np.ndarray:
        with ppl.maximum_a_posteriori():
            z, _ = self.partikel_crp(X.shape[1] - 1, torch.Size([X.shape[0]]))

        return z.cpu().numpy()


class ExpFamilyEvaluator(Evaluator):
    def __init__(self, model, num_particles):
        self.model = model
        self.num_particles = num_particles

        if torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False

        self.model.eval()
        if self.cuda:
            self.model.cuda()

    @classmethod
    def from_file(self, model_file: str, num_particles: int, four_param: bool):
        kwargs = {"in_dim": 512, "alpha": 1.0}
        if four_param:
            model = FourParameterExponentialFamilyParticleFilter(**kwargs)
        else:
            model = ExponentialFamilyParticleFilter(**kwargs)

        model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))

        return cls(model, num_particles)

    @classmethod
    def from_hypers(cls, hypers, num_particles: int):
        if hypers["likelihood"] == "NaturalFourParameterNormalInverseGammaLikelihood":
            model = FourParameterNoHurdleExponentialFamilyParticleFilter(
                in_dim=hypers["dim"], alpha=hypers["alpha"]
            )
            model.loc.data.fill_(hypers["loc"])
            model.log_mean_conc.data.fill_(math.log(hypers["mean_conc"]))
            model.log_conc.data.fill_(math.log(hypers["conc"]))
            model.log_scale.data.fill_(math.log(hypers["scale"]))
        else:
            raise ValueError

        return cls(model, num_particles)

    def nll_trace(self, X: np.ndarray, z: np.ndarray) -> np.ndarray:
        X_torch = torch.from_numpy(X)
        z_torch = torch.from_numpy(z).long()
        if self.cuda:
            X_torch = X_torch.cuda()
            z_torch = z_torch.cuda()
        with torch.no_grad():
            ret = self.model.nll_trace(X_torch, z_torch)
        return ret.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_torch = torch.from_numpy(X)
        if self.cuda:
            X_torch = X_torch.cuda()
        z_hat = []

        with torch.no_grad():
            ssm = self.model.construct_ssm()
            guide = self.model.construct_guide(
                ssm, X_torch, batch_shape=torch.Size([X_torch.shape[0]])
            )
            fk = GuidedFeynmanKac(
                guide, ssm, y_obs=X_torch, batch_shape=torch.Size([X_torch.shape[0]])
            )
            pf = ParticleFilter(fk, AdaptiveResampler(MultinomialResampler(), 0.5))
            result = pf.run(self.num_particles)
            best_inds = result.weights[..., -1].argmax(0)
            assert best_inds.ndim == 1
            z_hat.append(
                result.particles[best_inds, torch.arange(best_inds.shape[0])].cpu()
            )  # pyright: ignore

        z_hat = torch.concatenate(z_hat, 0)
        return z_hat.numpy()


@click.command
@click.argument("model_path")
@click.option(
    "--data_file", default="data/imagenet_crp_512d_100steps_1.0alpha_notrain.npz"
)
@click.option("--batch_size", default=100)
@click.option("--num_particles", default=100)
@click.option("--force", is_flag=True, default=False)
@click.option("--truncate", is_flag=True, default=False)
@click.option("--dry_run", is_flag=True, default=False)
@click.option("--seed", default=0)
def main(
    model_path: str,
    data_file: str,
    batch_size: int,
    num_particles: int,
    force: bool,
    truncate: bool,
    dry_run: bool,
    seed: int,
):
    torch.manual_seed(seed)

    data = np.load(data_file)
    in_dim = data["val_X"].shape[-1]
    assert data["test_X"].shape[-1] == in_dim

    if os.path.basename(model_path) == "circuit_model.pt":
        evaluator = GRUEvaluator(model_path, in_dim=in_dim)
        job_dir = os.path.dirname(model_path)
    elif os.path.basename(model_path) == "expfamily_model.pt":
        flags = torch.load(os.path.join(os.path.dirname(model_path), "flags.pt"))
        four_param = "four_param" in flags and flags["four_param"]
        evaluator = ExpFamilyEvaluator.from_file(
            model_path, num_particles, four_param=four_param
        )
        job_dir = os.path.dirname(model_path)
    else:
        identifier = os.path.basename(model_path).split("_")[-1]
        if identifier.isnumeric():
            alpha = float(identifier)
            evaluator = CRPEvaluator(alpha)
            job_dir = model_path
        else:
            dataset_name = "_".join(os.path.basename(model_path).split("_")[:-1])
            if dataset_name == "synthetic_normal_inverse_gamma_data":
                true_hypers = {
                    "alpha": 1.0,
                    "dim": 2,
                    "loc": 0.0,
                    "mean_conc": 0.01,
                    "conc": 2.0,
                    "scale": 2.0,
                    "likelihood": "NaturalFourParameterNormalInverseGammaLikelihood",
                }
                evaluator = ExpFamilyEvaluator.from_hypers(true_hypers, num_particles)
                job_dir = model_path
            else:
                raise ValueError

    def is_missing(file_name: str) -> bool:
        return not os.path.isfile(os.path.join(job_dir, file_name))

    def evaluate_nll(split: str):
        X = data[f"{split}_X"]
        z = data[f"{split}_z"]

        if (
            force
            or is_missing(f"{split}_nll_trace.npy")
            or is_missing(f"{split}_benchmark.npz")
        ):
            start = time.time()
            nll_trace = extract_nll_trace(evaluator, X, z, batch_size)
            elapsed = time.time() - start
            if not dry_run:
                np.save(os.path.join(job_dir, f"{split}_nll_trace.npy"), nll_trace)
                np.savez(
                    os.path.join(job_dir, f"{split}_benchmark.npz"),
                    elapsed=elapsed,
                    num_examples=X.shape[0],
                    batch_size=batch_size,
                )

    def evaluate_clustering(split: str):
        X = data[f"{split}_X"]
        z = data[f"{split}_z"]

        if truncate:
            X = X[:batch_size]
            z = z[:batch_size]

        if force or is_missing(f"{split}_clustering.npz"):
            clustering_results = extract_clustering_results(
                evaluator,
                X,
                z,
                batch_size,
            )
            if not dry_run:
                np.savez(
                    os.path.join(job_dir, f"{split}_clustering.npz"),
                    **clustering_results,
                )

    print("nll val")
    evaluate_nll("val")
    print("clustering val")
    evaluate_clustering("val")
    print("nll test")
    evaluate_nll("test")
    print("clustering test")
    evaluate_clustering("test")


if __name__ == "__main__":
    # import cProfile
    # cProfile.run("main()", "profile_results.prof")
    main()
