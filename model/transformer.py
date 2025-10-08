import torch

import torch.nn as nn

from blocks.encoder import Encoder
from blocks.decoder import Decoder

from embedding import Embedding, PositionalEmbedding
from generator import Generator

class Transformer(nn.Module):

    def __init__(self, source_len, target_len, num_layer: int = 6, max_len: int = 4096, num_heads: int = 8, dim_model: int = 512, attn_dropout: float | None = None, 
                 linear_dim: int = 2048, linear_dropout: float | None = None, pos_embed_dropout: float | None = None):
        
        self.encoder_embed = nn.Sequential(
            Embedding(vocab_len=source_len, dim_model=dim_model),
            PositionalEmbedding(dim_model=dim_model, dropout=0.1, max_len=max_len)
        )

        self.decoder_embed = nn.Sequential(
            Embedding(vocab_len=target_len, dim_model=dim_model),
            PositionalEmbedding(dim_model=dim_model, dropout=0.1, max_len=max_len)
        )

        self.encoder = Encoder(num_layer=num_layer, num_heads=num_heads, dim_model=dim_model,
                               attn_dropout=attn_dropout, linear_dim=linear_dim, linear_dropout=linear_dropout)
        
        self.decoder = Decoder(num_layer=num_layer, num_heads=num_heads, dim_model=dim_model,
                               attn_dropout=attn_dropout, linear_dim=linear_dim, linear_dropout=linear_dropout)
        
        self.generator = Generator(vocab_len=target_len, dim_model=dim_model)

    
    def encode(self, source: torch.tensor, mask: torch.tensor)->torch.tensor:
        return self.encoder.forward(input=source, mask=mask)
    
    def decode(self, target: torch.tensor, decoder_mask: torch.tensor, 
               memory: torch.tensor, memory_mask: torch.tensor)->torch.tensor:
        
        return self.decoder.forward(input=target, decoder_mask=decoder_mask, memory=memory, memory_mask=memory_mask)
    
    def forward(self, source: torch.tensor, target: torch.tensor, decoder_mask: torch.tensor, 
               memory: torch.tensor, memory_mask: torch.tensor)->torch.tensor:
        
        encoder_output = self.encode(input=self.encoder_embed(input=source), mask=memory_mask)
        decoder_output = self.decode(input=self.decoder_embed(input=target), decoder_mask=decoder_mask, memory=encoder_output, memory_mask=memory_mask)

        output = self.generator(decoder_output)

        del encoder_output
        del decoder_output

        return output