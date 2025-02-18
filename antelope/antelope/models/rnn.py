from absl import flags

import torch
import torch.nn as nn
import torch.nn.functional as F

FLAGS = flags.FLAGS
flags.DEFINE_integer("hidden_size", 100, "lstm hidden size")
flags.DEFINE_integer("num_layers", 1, "number of rnn layers")


class GRU(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_size: int,
        num_layers: int,
        max_clusters=None,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            in_dim + out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, out_dim)
        self.h_init = nn.Parameter(0.01 * torch.randn(self.num_layers, 1, hidden_size))
        self.max_clusters = max_clusters

    @classmethod
    def from_flags(cls, in_dim: int, out_dim: int, **kwargs):
        return cls(
            in_dim,
            out_dim,
            hidden_size=FLAGS.hidden_size,
            num_layers=FLAGS.num_layers,
            **kwargs
        )

    def mask(self, logits, y_onehot_shift):
        mask = ~torch.cummax(y_onehot_shift, dim=-2).values.bool()
        mask_shift = F.pad(mask, (1, 0), mode="constant")[:, :, :-1]

        if self.max_clusters is not None:
            mask_shift[:, :, self.max_clusters :] = True

        return logits.masked_fill(mask_shift, -float("inf"))

    def forward_init(self, x: torch.Tensor):
        # x: batch_size x in_dim
        N = x.size(0)
        input_features = torch.cat(
            [x, torch.zeros(N, self.out_dim, device=x.device)], -1
        )
        z, h = self.rnn(
            input_features.unsqueeze(1),
            self.h_init.expand(self.num_layers, N, self.hidden_size).contiguous(),
        )
        logits = self.mask(
            self.fc(F.relu(z)), torch.zeros(N, 1, self.out_dim, device=x.device)
        ).squeeze(1)
        prev_max_class = -torch.ones(N, device=x.device).long()
        return logits, prev_max_class, h

    def forward_step(
        self,
        x: torch.Tensor,
        y_prev: torch.Tensor,
        prev_max_class: torch.Tensor,
        h: torch.Tensor,
    ):
        # x.shape: torch.Size([batch_size, self.in_dim])
        # y_prev.shape: torch.Size([batch_size])
        # prev_max_class.shape: torch.Size([batch_size])
        # h.shape: torch.Size([self.num_layers, batch_size, self.hidden_size])

        input_features = torch.cat([x, F.one_hot(y_prev, num_classes=self.out_dim)], -1)

        prev_max_class = torch.maximum(y_prev, prev_max_class)

        z, h = self.rnn(input_features.unsqueeze(1), h)

        max_mask = F.one_hot(prev_max_class, self.out_dim)
        max_mask = 1 - torch.cummax(max_mask, -1).values + max_mask

        logits = self.mask(self.fc(F.relu(z)), max_mask.unsqueeze(1)).squeeze(1)
        return logits, prev_max_class, h

    def predict(self, x: torch.Tensor):
        # Predict the MAP trajectory according to the model
        # x.shape: torch.Size([batch_size, sequence_length self.in_dim])
        sequence_length = x.size(1)

        predictions = []
        cur_logits, prev_max_class, h = self.forward_init(x[:, 0])
        predictions.append(torch.argmax(cur_logits, -1))

        for t in range(1, sequence_length):
            cur_logits, prev_max_class, h = self.forward_step(
                x[:, t], predictions[-1], prev_max_class, h
            )
            predictions.append(torch.argmax(cur_logits, -1))

        return torch.stack(predictions, -1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        N = x.size(0)
        y_onehot_shift = F.pad(
            F.one_hot(y, num_classes=self.out_dim), (0, 0, 1, 0), mode="constant"
        )[:, :-1, :]
        input_features = torch.cat([x, y_onehot_shift], -1)
        z, _ = self.rnn(
            input_features,
            self.h_init.expand(self.num_layers, N, self.hidden_size).contiguous(),
        )
        logits = self.fc(F.relu(z))

        return self.mask(logits, y_onehot_shift)

    def condition(self, x: torch.Tensor, y: torch.Tensor):
        N = x.size(0)
        y_onehot_shift = F.pad(
            F.one_hot(y, num_classes=self.out_dim), (0, 0, 1, 0), mode="constant"
        )[:, :-1, :]
        input_features = torch.cat([x, y_onehot_shift], -1)
        _, h = self.rnn(
            input_features, self.h_init.expand(self.num_layers, N, self.hidden_size)
        )
        return h
