import torch

import torch.nn as nn

from attention.multihead_attn import MultiHeadAttn


class DecoderLayer(nn.Module):
    def __init__(self, num_heads: int, dim_model: int, linear_dim: int, 
                 attn_dropout: float | None = None, linear_dropout: float | None = None):
        
        """
        linear_dim: dimension of the hidden layer
        linear_dropout: dropout rate in the feed-forward network
        """

        super().__init__()

        self.self_attn = MultiHeadAttn(num_heads=num_heads, dim_model=dim_model, dropout=attn_dropout)
        self.cross_attn = MultiHeadAttn(num_heads=num_heads, dim_model=dim_model, dropout=attn_dropout)

        self.self_attn_norm = nn.LayerNorm(normalized_shape=dim_model)
        self.cross_attn_norm = nn.LayerNorm(normalized_shape=dim_model)

        self.linear_layer_norm = nn.LayerNorm(normalized_shape=dim_model)

        self.linear = nn.Sequential(
                    nn.Linear(dim_model, linear_dim),
                    nn.LeakyReLU(),
                )
        
        if linear_dropout:
            self.linear.append(nn.Dropout(linear_dropout))
        
        self.linear.append(nn.Linear(linear_dim, dim_model))
    
    def forward(self, input: torch.tensor, memory: torch.tensor, 
                memory_mask: torch.tensor, decoder_mask:torch.tensor)->torch.tensor:
        # Memory is the ouput of the encoder
        # Memory_mask is the mask we use in the encoder
        # Input is the input of the decoder: ground truth during training, previously predicted values during inference

        self_attn_output = self.self_attn.forward(key=input, query=input, value=input, mask=decoder_mask)
        self_attn_output = self.self_attn_norm.forward(self_attn_output+input)

        cross_attn_output = self.cross_attn.forward(key=memory, query=self_attn_output, value=memory, mask=memory_mask)
        cross_attn_output = self.cross_attn_norm.forward(cross_attn_output+self_attn_output)


        linear_output = self.linear(cross_attn_output)
        linear_output = self.linear_layer_norm.forward(linear_output+cross_attn_output)

        del self_attn_output
        del cross_attn_output

        return linear_output
    

class Decoder(nn.Module):

    def __init__(self, num_layer: int = 6, num_heads: int = 8, dim_model: int = 512, attn_dropout: float | None = None, 
                 linear_dim: int = 2048, linear_dropout: float | None = None):
        
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(num_heads=num_heads, dim_model=dim_model, attn_dropout=attn_dropout,
                                    linear_dim=linear_dim, linear_dropout=linear_dropout) for layer in range(num_layer)])
        
    def forward(self, input: torch.tensor, memory:torch.tensor, decoder_mask:torch.tensor, 
                memory_mask:torch.tensor)->torch.tensor:
        
        output = input
        for layer in self.layers:
            output = layer.forward(input=output, memory=memory, decoder_mask=decoder_mask, memory_mask=memory_mask)

        return output



