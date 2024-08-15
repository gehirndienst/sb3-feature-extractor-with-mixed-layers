import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RnnAttention(nn.Module):
    def __init__(self, method: str, hidden_size: int, device: th.device = th.device("cpu")) -> None:
        super(RnnAttention, self).__init__()
        self.device = device

        self.method = method.lower()
        self.hidden_size = hidden_size

        # conatenate context vector and hidden state
        self.concat_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if self.method == "general":
            self.attn = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, rnn_output: th.Tensor, rnn_state: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # assume RNN was instantiated with batch_first=True, so that
        # rnn_outputs.shape = (batch_size, sequence_length, hidden_size),
        # rnn_state.shape = (batch_size, num_directions * hidden_size) [we took the last one]
        if self.method == "dot":
            weights = th.matmul(rnn_output, rnn_state.unsqueeze(2))
        elif self.method == "general":
            weights = self.attn(rnn_output)
            weights = th.matmul(weights, rnn_state.unsqueeze(2))

        weights = F.softmax(weights.squeeze(2), dim=1)
        context = th.matmul(rnn_output.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)
        hidden = th.matmul(self.concat_linear(th.cat((context, rnn_state), dim=1)))

        return hidden, weights
