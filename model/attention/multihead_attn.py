
from torch import nn
import torch


class MultiHeadAttn(nn.module):
    """
    Implement the multihead attention based on the dot-product attention in Attention is All Your Need.
    """

    def __init__(self, num_heads: int, dim_model: int, dropout: float | None = None):
        # We assume that d_key = d_value
        # dim_model: the dimension of embedding vectors
        super().__init__()

        assert dim_model%num_heads == 0, f"{dim_model} is not divisible by {num_heads}"
        self.num_heads = num_heads
        self.dim_model = dim_model

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        # Totally 3 linear layers for Q, K, and V. Another linear layer for concat.
        self.linear = [nn.linear(dim_model, dim_model) for i in range(4)]

        self.weights = None

    def forward(self, query: torch.tensor, 
                key: torch.tensor, 
                value: torch.tensor, 
                mask: torch.tensor | None = None)->torch.tensor:
        
        # query.shape = (Batch, d_sequence, d_model)
        # mask.shape = (1, d_model, d_model)
        n_batch = query.shape(dim = 0)
        
        # Slice the projected matrices into num_head parts
        query, key, value = (
            linear(x).view(n_batch, -1, self.num_heads, self.dim_model//self.num_heads)\
                            .transpose(1, 2) for linear, x in zip(self.linear[:-1], (query, key, value))
        )

        if mask is not None:
            mask.unsqueeze(1)

        multi_attn, self.weights = self.self_attn(
                        query=query,
                        key=key,
                        value=value,
                        mask=mask,
                        dropout=self.dropout
                    )
        
        # Concatenate outputs of attention heads
        multi_attn = multi_attn.transpose(1,2).contiguous().view(n_batch, -1, self.dim_model)

        # Save memory
        del query, key, value

        return self.linear[-1](multi_attn)



    def self_attn(self, query: torch.tensor, 
                  key: torch.tensor, 
                  value: torch.tensor, 
                  mask: torch.tensor | None = None,
                  dropout: nn.Dropout | None = None) -> torch.tensor:
        """
        output = mask(softmax(Q(K^T)/sqrt_{d_key}))V
        """
        weights = nn.softmax(torch.matmul(query, key.transpose(-2, -1))/torch.sqrt(self.dim_model//self.num_heads))

        if mask is not None:
            # Assign -inf to masked positions so the their values become 0 after softmax.
            weights = weights.mask_filled(mask == 0, -1e9)
        
        if dropout is not None:
            weights = dropout(weights)

        return torch.matmul(weights, value), weights

    
