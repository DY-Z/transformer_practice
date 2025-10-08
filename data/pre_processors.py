from collections.abc import Iterable, Generator

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import spacy
from torchtext.vocab import build_vocab_from_iterator

def split(data: pd.DataFrame, save_dir: str, train_size: float = 0.7, test_size: float = 0.2)->None:
    
    """
    Split the dataset into train/val/test folders.

    data_dir: the local directory to access the entire dataset
    save_dir: the local directory to save the dataset
    train_size: the proportion of training data
    test_size: the proportion of testing data
    """

    assert train_size + test_size < 1.0, "The sum of train_ratio and test_ratio should be smaller than 1."

    train, test = train_test_split(data, train_size=train_size)

    test.reset_index(drop=True).to_csv(save_dir+"/test.csv", index=False)
    del test

    train, val = train_test_split(train, train_size=train_size/(1-test_size))

    train.reset_index(drop=True).to_csv(save_dir+"/train.csv", index=False)
    val.reset_index(drop=True).to_csv(save_dir+"/val.csv", index=False)

    del train
    del val
    


def yield_tokens(data_iter: Iterable, tokenizer: spacy.language.Language)->Generator:

    #print(data_iter)
    #TODO: Check whether there are NaNs in data. If so, remove them. Or probably we can convert them to strings?
    for data in data_iter:
        #print(data)
        assert isinstance(data, str), "Raw data should be strings."
        yield [token.text for token in tokenizer.tokenizer(data)]



def build_source_target_vocab(raw: pd.DataFrame, source_tokenizer: spacy.language.Language, 
                              target_tokenizer: spacy.language.Language, save_dir: str)->None:
    
    # Assume that raw contains two different languages.
    # Num of raw's columns should be same as the num of tokenizers

    columns = raw.columns
    assert len(columns)==2, "Types of languages should be the same as the number of tokenizers."

    source, target = columns

    source_vocab = build_vocab_from_iterator(
        iterator=yield_tokens(data_iter=raw[source],tokenizer=source_tokenizer),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
        special_first=True
    )

    target_vocab = build_vocab_from_iterator(
        iterator=yield_tokens(data_iter=raw[target],tokenizer=target_tokenizer),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
        special_first=True
    )

    source_vocab.set_default_index(source_vocab["<unk>"])
    target_vocab.set_default_index(target_vocab["<unk>"])

    torch.save((source_vocab, target_vocab), save_dir+"/vocab.pt")


def load_vocab(load_dir: str)->Iterable:

    vocab_src, vocab_tgt = torch.load(load_dir+"/vocab.pt")

    return vocab_src, vocab_tgt


        





    




