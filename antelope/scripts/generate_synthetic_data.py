import torch
import numpy as np

from partikel.exponential_family import (
    NormalInverseGamma,
    NaturalFourParameterNormalInverseGammaPrior,
    NaturalFourParameterNormalInverseGammaLikelihood,
)
from partikel.markov_process import ChineseRestaurantProcess
from partikel.state_space_model import CollapsedInfiniteMixtureModel


def main():
    alpha = 1.0
    dim = 2

    loc = 0.0
    mean_conc = 0.01
    conc = 2.0
    scale = 2.0

    nig_ordinary = NormalInverseGamma(
        loc=torch.full((dim,), loc),
        mean_concentration=torch.full((dim,), mean_conc),
        concentration=torch.full((dim,), conc),
        scale=torch.full((dim,), scale),
    )

    conjugate_prior = NaturalFourParameterNormalInverseGammaPrior.from_ordinary(
        nig_ordinary
    )
    likelihood = NaturalFourParameterNormalInverseGammaLikelihood()

    # construct state-space model
    ssm = CollapsedInfiniteMixtureModel(
        ChineseRestaurantProcess(alpha), conjugate_prior, likelihood
    )

    def generate_split(ssm, num_examples: int, batch_size: int):
        i = 0
        X_all = []
        z_all = []
        while i < num_examples:
            print(".", end="")
            batch_shape = torch.Size([min(batch_size, (num_examples - i))])
            (z, X), _ = ssm(99, sample_shape=batch_shape)
            X_all.append(X)
            z_all.append(z)
            i += batch_shape[0]

        X_all = torch.concat(X_all, 0)
        z_all = torch.concat(z_all, 0)
        return z_all, X_all

    train_z, train_X = generate_split(ssm, 100000, 100)
    val_z, val_X = generate_split(ssm, 10000, 100)
    test_z, test_X = generate_split(ssm, 10000, 100)

    np.savez(
        "synthetic_normal_inverse_gamma_data.npz",
        train_X=train_X.numpy(),
        train_z=train_z.numpy(),
        val_X=val_X.numpy(),
        val_z=val_z.numpy(),
        test_X=test_X.numpy(),
        test_z=test_z.numpy(),
    )


if __name__ == "__main__":
    main()
