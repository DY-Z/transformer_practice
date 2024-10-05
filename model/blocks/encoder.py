import torch

import torch.nn as nn

from attention.multihead_attn import MultiHeadAttn

class EncoderLayer(nn.Module):
    def __init__(self, num_heads: int, dim_model: int, linear_dim: int, 
                 attn_dropout: float | None = None, linear_dropout: float | None = None):

        """
        linear_dim: dimension of the hidden layer
        linear_dropout: dropout rate in the feed-forward network
        """

        super().__init__()

        self.multihead_attn = MultiHeadAttn(num_heads=num_heads, dim_model=dim_model, dropout=attn_dropout)

        self.attn_layer_norm = nn.LayerNorm(normalized_shape=dim_model)
        self.linear_layer_norm = nn.LayerNorm(normalized_shape=dim_model)

        self.linear = nn.Sequential(
                    nn.Linear(dim_model, linear_dim),
                    nn.LeakyReLU(),
                )
        
        if linear_dropout:
            self.linear.append(nn.Dropout(linear_dropout))
        
        self.linear.append(nn.Linear(linear_dim, dim_model))
        
    def forward(self, input: torch.tensor, mask: torch.tensor)->torch.tensor:

        attn_output = self.multihead_attn.forward(key=input, query=input, value=input, mask=mask)
        attn_output = self.attn_layer_norm(attn_output+input)

        linear_output = self.linear(attn_output)
        linear_output = self.linear_layer_norm(linear_output+attn_output)

        del attn_output

        return linear_output

class Encoder(nn.Module):
    
    def __init__(self, num_layer: int = 6, num_heads: int = 8, dim_model: int = 512, attn_dropout: float | None = None, 
                 linear_dim: int = 2048, linear_dropout: float | None = None):
        
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(num_heads=num_heads, dim_model=dim_model, attn_dropout=attn_dropout,
                                                 linear_dim=linear_dim, linear_dropout=linear_dropout) for layer in range(num_layer)])
        

    def forward(self, input: torch.tensor, mask: torch.tensor)->torch.tensor:

        output = input
        for layer in self.layers:
            output = layer.forward(output)

        return output
        