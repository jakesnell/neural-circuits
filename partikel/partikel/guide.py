import torch
from torch.distributions import Categorical

from partikel.state_space_model import CollapsedIMMStats, CollapsedInfiniteMixtureModel
from partikel import ppl


class CollapsedInfiniteMixtureGuide:
    def __init__(
        self,
        ssm: CollapsedInfiniteMixtureModel,
        y_obs: torch.Tensor,
        batch_shape: torch.Size,
    ):
        self.ssm = ssm
        self.y_obs = y_obs  # batch_shape, time, event_shape
        self.batch_shape = batch_shape

    def __len__(self):
        return self.y_obs.size(len(self.batch_shape))

    def init(self, sample_shape: torch.Size = torch.Size([])):
        self.ssm.prior.init(sample_shape + self.batch_shape)

    def step(
        self,
        t: int,
        stats: CollapsedIMMStats,
        sample_shape: torch.Size = torch.Size([]),
    ):
        prior_log_prob = torch.log_softmax(
            self.ssm.prior.predictive_distribution(  # pyright: ignore
                stats.prior_stats
            ).logits(self.ssm.max_clusters),
            -1,
        )
        y_t = torch.select(self.y_obs, len(self.batch_shape), t)
        y_t = y_t.expand(sample_shape + y_t.shape)
        y_t = y_t.unsqueeze(prior_log_prob.ndim - 1)

        log_likelihood = self.ssm.likelihood.predictive_log_prob_closure(
            stats.posterior
        )(y_t)
        log_likelihood = log_likelihood.sum(
            tuple(range(log_likelihood.ndim))[prior_log_prob.ndim :]
        )
        posterior = Categorical(logits=prior_log_prob + log_likelihood)
        ppl.sample(f"x_{t}", posterior)

    def trace_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[: len(self.batch_shape)] == self.batch_shape
        assert x.size(len(self.batch_shape)) == len(self)

        observed_kwargs = {
            f"y_{t}": torch.select(self.y_obs, len(self.batch_shape), t)
            for t in range(len(self))
        }
        observed_kwargs.update(
            {
                f"x_{t}": torch.select(x, len(self.batch_shape), t)
                for t in range(len(self))
            }
        )

        with ppl.observed(**observed_kwargs):
            with ppl.trace() as guide_trace:
                self.init()

            with ppl.replay(guide_trace):
                _, stats = self.ssm.init(self.batch_shape)

            for t in range(1, len(self)):
                with ppl.trace(guide_trace):
                    self.step(t, stats)

                _, stats = self.ssm.step(t, stats)

        log_probs = ppl.trace_log_prob(
            guide_trace, sample_ndims=len(self.batch_shape), reduce_structure=False
        )

        return torch.stack(
            [log_probs[f"x_{t}"] for t in range(len(self))], -1  # pyright: ignore
        )
