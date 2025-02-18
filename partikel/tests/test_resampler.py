import torch
from torch.testing import assert_close

from partikel.particle_set import ParticleSet
from partikel.resampler import MultinomialResampler, AdaptiveResampler, gather_ancestors


def test_multinomial_resampler():
    num_particles = 7
    batch_size = 500

    logits = torch.randn(num_particles, batch_size)
    loc = torch.rand(num_particles, batch_size, 2)
    ps = ParticleSet(logits, loc)

    resampler = MultinomialResampler()

    ps_new, ancestors = resampler.resample(ps)

    # shape of weights should not have changed
    assert ps_new.weights.shape == ps.weights.shape
    # shape of particles should not have changed
    assert ps_new.particles.shape == ps.particles.shape

    # should not have any ancestors beyond the number of particles
    assert torch.all(ancestors < num_particles)

    assert ancestors.shape == ps.weights.shape

    assert_close(ps_new.particles, gather_ancestors(ancestors, loc))


def test_adaptive_resampler():
    num_particles = 7
    batch_size = 500

    logits = torch.randn(num_particles, batch_size)
    loc = torch.rand(num_particles, batch_size, 2)
    ps = ParticleSet(logits, loc)

    resampler = AdaptiveResampler(MultinomialResampler(), 0.5)

    ps_new, ancestors = resampler.resample(ps)

    # shape of weights should not have changed
    assert ps_new.weights.shape == ps.weights.shape
    # shape of particles should not have changed
    assert ps_new.particles.shape == ps.particles.shape

    # should not have any ancestors beyond the number of particles
    assert torch.all(ancestors < num_particles)

    assert ancestors.shape == ps.weights.shape

    assert_close(ps_new.particles, gather_ancestors(ancestors, loc))


def test_adaptive_resampler():
    num_particles = 7
    batch_size = 500

    logits = torch.randn(num_particles, batch_size)
    loc = torch.rand(num_particles, batch_size, 2)
    ps = ParticleSet(logits, loc)

    resampler = AdaptiveResampler(MultinomialResampler(), 0.0)

    ps_new, ancestors = resampler.resample(ps)

    # shape of weights should not have changed
    assert ps_new.weights.shape == ps.weights.shape
    # shape of particles should not have changed
    assert ps_new.particles.shape == ps.particles.shape

    # should not have any ancestors beyond the number of particles
    assert torch.all(ancestors < num_particles)

    assert ancestors.shape == ps.weights.shape

    assert_close(ps_new.particles, gather_ancestors(ancestors, loc))

    assert_close(
        torch.arange(num_particles).unsqueeze(-1).expand(num_particles, batch_size),
        ancestors,
    )
