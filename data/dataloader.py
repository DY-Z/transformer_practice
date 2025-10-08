import torch
import pandas as pd
from typing import (
    Callable, 
    Any,
    Iterable,
    Tuple
)

import spacy

from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab

from pre_processors import yield_tokens


class SmallTranslationDataset(Dataset):

    def __init__(self, data_dir: str, transform: Callable | None):

        # Assume the dataset is small.
        # Load the entire dataset as it can fit into the memory.
        # Only accept csv files here.
        super().__init__()

        assert data_dir.endswith(".csv")

        self.data = pd.read_csv(data_dir)
        self.transform = transform

    def __len__(self)->int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Any:
        return self.data.iloc[index]
    

def collect_batch(batch: Iterable, source_tokenizer: spacy.language.Language, 
                  target_tokenizer: spacy.language.Language, source_vocab: Vocab, target_vocab: Vocab,
                  device: Any, max_padding: int = 128, pad_id: int = 2)->Tuple[torch.Tensor]:
    
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id

    source_list, target_list = [], []
    for (source, target) in batch:
        processed_source = torch.cat(
            [
                bs_id,
                torch.tensor(
                    source_vocab(yield_tokens(data_iter=source, tokenizer=source_tokenizer)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_target = torch.cat(
            [
                bs_id,
                torch.tensor(
                    target_vocab(yield_tokens(data_iter=target, tokenizer=target_tokenizer)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )

        source_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_source,
                (
                    0,
                    max_padding - len(processed_source),
                ),
                value=pad_id,
            )
        )
        target_list.append(
            pad(
                processed_target,
                (0, max_padding - len(processed_target)),
                value=pad_id,
            )
        )

    source = torch.stack(source_list)
    target = torch.stack(target_list)
    return (source, target)


def create_single_gpu_dataloader(batch_size: int, source_tokenizer: spacy.language.Language, 
                  target_tokenizer: spacy.language.Language, source_vocab: Vocab, target_vocab: Vocab,
                  device: Any, max_padding: int = 128, pad_id: int = 2):
    pass
    



class Batch: 
    """Object for holding a batch of data with mask during training.""" 
    def __init__(self, source: torch.Tensor, target: torch.Tensor | None = None, pad:int=2): # 2 = <blank>
        # source: The input 2D tensor of tokens with the shape (batch, len_sentence_source)
        # target: The ground-truth 2D tensor of tokens with the shape (batch, len_sentence_target)
        self.src = source
        self.src_mask = (source != pad).unsqueeze(-2) 
        if target is not None: 
            self.tgt = target[:, :-1] 
        self.tgt_y = target[:, 1:] 
        self.tgt_mask = self.make_std_mask(self.tgt, pad) 
        self.ntokens = (self.tgt_y != pad).data.sum() 
        
    @staticmethod 
    def make_std_mask(tgt: torch.Tensor, pad:int): 
        "Create a mask to hide padding and future words." 
        tgt_mask = (tgt != pad).unsqueeze(-2) 
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data) 
        return tgt_mask

def subsequent_mask(size: int):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0
