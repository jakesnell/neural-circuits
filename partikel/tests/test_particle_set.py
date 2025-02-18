import torch
from torch.testing import assert_close

from partikel.particle_set import ParticleSet
from partikel.util import stack_structure


def test_init_simple():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    p = ParticleSet(logits=torch.zeros(2), particles=x)
    assert_close(p.estimate(), torch.tensor([2.0, 3.0]))


def test_init_composite():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[-2.0, 1.0], [4.0, 5.0]])
    p = ParticleSet(logits=torch.zeros(2), particles=[x, y])

    est = p.estimate()
    assert len(est) == 2  # pyright: ignore
    assert_close(est[0], torch.tensor([2.0, 3.0]))  # pyright: ignore
    assert_close(est[1], torch.tensor([1.0, 3.0]))  # pyright: ignore


def test_stack_simple():
    t = 5
    particles = []
    particle_sets = []
    for i in range(t):
        x = torch.arange(i, i + 4).view(2, 2)
        particles.append(x)
        particle_sets.append(ParticleSet(logits=torch.zeros(2), particles=x))

    p = stack_structure(particle_sets, 1)
    assert_close(p.particles, torch.stack(particles, 1))  # pyright: ignore


def test_stack_composite():
    t = 5
    xs = []
    ys = []
    particle_sets = []
    for i in range(t):
        x = torch.arange(i, i + 4).view(2, 2)
        y = torch.arange(2 * i, 2 * i + 4).view(2, 2)
        xs.append(x)
        ys.append(y)
        particle_sets.append(ParticleSet(logits=torch.zeros(2), particles=[x, y]))

    p = stack_structure(particle_sets, 1)
    assert_close(p.particles[0], torch.stack(xs, 1))  # pyright: ignore
    assert_close(p.particles[1], torch.stack(ys, 1))  # pyright: ignore
