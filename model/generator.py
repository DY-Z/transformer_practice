import torch

import torch.nn as nn

from torch.nn.functional import log_softmax


class Generator(nn.Module):
    """
    Generate the distribution of the token based on the decoder's output.

    One linear layer + one softmax layer
    """

    def __init__(self, vocab_len: int, dim_model: int = 512):
        super().__init__()

        self.linear = nn.Linear(dim_model, vocab_len)
    
    def forward(self, input: torch.tensor)->torch.tensor:
        # The input is in shape of (d_sequence, vocab_len) for each batch.
        # We apply log_softmax along the last dimension.
        return log_softmax(input=self.linear(input), dim=-1)
    
class GreedyDecoder:
    pass

