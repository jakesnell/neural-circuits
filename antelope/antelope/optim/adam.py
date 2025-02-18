from absl import flags
import torch.optim as optim

flags.DEFINE_float("lr", 1e-3, "learning rate")


def adam(params):
    return optim.Adam(params, lr=flags.FLAGS.lr)
