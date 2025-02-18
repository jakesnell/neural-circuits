import torch
from torch.testing import assert_close

from antelope.samplers.crp_batch_sampler import CRPBatchSampler
from antelope.models.rnn import GRU


def generate_sample_data_and_model(
    in_dim, out_dim, batch_size, sequence_length, num_clusters, hidden_size
):
    # generate some data
    cluster_centers = torch.randn(num_clusters, in_dim) * 3
    num_examples_per_cluster = 50

    cluster_data = [
        cluster_centers[i] + 0.1 * torch.randn(num_examples_per_cluster, in_dim)
        for i in range(num_clusters)
    ]
    sampler = CRPBatchSampler(cluster_data, 1.0, batch_size, sequence_length)
    X, z = sampler.sample()

    model = GRU(
        in_dim, out_dim=out_dim, hidden_size=hidden_size, num_layers=2, max_clusters=100
    )
    return (X, z), model


def test_foward():
    in_dim = 2
    out_dim = 10
    batch_size = 3
    sequence_length = 6
    num_clusters = 7
    hidden_size = 27

    (X, z), model = generate_sample_data_and_model(
        in_dim, out_dim, batch_size, sequence_length, num_clusters, hidden_size
    )

    assert X.shape == torch.Size([batch_size, sequence_length, in_dim])
    assert z.shape == torch.Size([batch_size, sequence_length])

    logits = model(X, z)
    assert logits.shape == torch.Size([batch_size, sequence_length, out_dim])


def test_step():
    in_dim = 2
    out_dim = 10
    batch_size = 3
    sequence_length = 6
    num_clusters = 7
    hidden_size = 27

    # X: batch_size, sequence_length, in_dim
    # z: batch_size, sequence_length
    (X, z), model = generate_sample_data_and_model(
        in_dim, out_dim, batch_size, sequence_length, num_clusters, hidden_size
    )

    logits = model(X, z)

    cur_logits, prev_max_class, h = model.forward_init(X[:, 0, :])
    assert cur_logits.shape == torch.Size([batch_size, out_dim])
    assert prev_max_class.shape == torch.Size([batch_size])
    assert torch.all(prev_max_class == -1)
    assert h.shape == torch.Size([model.num_layers, batch_size, hidden_size])
    assert_close(logits[:, 0], cur_logits)

    for i in range(1, sequence_length):
        cur_logits, prev_max_class, h = model.forward_step(
            X[:, i], z[:, i - 1], prev_max_class, h
        )
        assert cur_logits.shape == torch.Size([batch_size, out_dim])
        assert h.shape == torch.Size([model.num_layers, batch_size, hidden_size])

        assert_close(logits[:, i], cur_logits)


def test_predict():
    in_dim = 2
    out_dim = 10
    batch_size = 3
    sequence_length = 6
    num_clusters = 7
    hidden_size = 27

    # X: batch_size, sequence_length, in_dim
    # z: batch_size, sequence_length
    (X, z), model = generate_sample_data_and_model(
        in_dim, out_dim, batch_size, sequence_length, num_clusters, hidden_size
    )

    z_hat = model.predict(X)
    assert z_hat.shape == z.shape
