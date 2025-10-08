import torch

import torch.nn as nn
from torchtext.vocab import Vocab

import math

class PositionalEmbedding(nn.Module):
    # Positional embedding using sine and cosine.
    # embed_{pos, 2i} = sin(pos/10000^{2i/dim_model})
    # embed_{pos, 2i+1} = cos(pos/10000^{2i/dim_model})

    def __init__(self, dim_model: int = 512, dropout: float | None = None, max_len: int = 4096):
        # max_len is the max number of tokens in one sentence.
        super().__init__()

        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

        # construct positional embedding
        pos_embed = torch.zeros(max_len, dim_model)

        position = torch.arange(0, max_len).unsqueeze(0) # for position OUT_PRODUCT wave
        wave = torch.exp(
            torch.arange(0, dim_model, 2)*(-1.0)*math.log(10000.0)/dim_model # for numerical stability
        )

        pos_embed[:, 0::2] = torch.sin(position*wave)
        pos_embed[:, 1::2] = torch.cos(position*wave)

        # We should expand pos_embed to the shape (1, max_len, dim_model) for broadcasting
        pos_embed = pos_embed.unsqueeze(0)
        self.register_buffer("pos_embed", pos_embed)

    def forward(self, input: torch.tensor)->torch.tensor:

        output = input+self.pos_embed[:, :input.size(1)].requires_grad_(False)

        if self.dropout:
            output = self.dropout(output)

        return output

class Embedding(nn.Module):
    
    def __init__(self, vocab_len: int, dim_model: int = 512):
        # TODO: check the data type of vocab. Tensor or Vocab?
        super().__init__()
        self.embed = nn.Embedding(vocab_len, dim_model)
        self.dim_model = dim_model

    def forward(self, input: torch.tensor)->torch.tensor:
        return self.embed(input) * math.sqrt(self.dim_model)
