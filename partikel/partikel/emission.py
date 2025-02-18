from torch.distributions import Distribution, Normal
from partikel.stats import MarkovStats


class GaussianEmission:
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, stats: MarkovStats) -> Distribution:
        return Normal(stats.loc, self.sigma)
