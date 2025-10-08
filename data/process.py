import torch
import spacy

import pandas as pd

from pre_processors import (
    split,
    build_source_target_vocab,
    load_vocab,
    yield_tokens
)
from sklearn.model_selection import train_test_split

def preprocess():

    data_dir = "/mnt/c/Users/dzhon/Desktop/myProjects/data/en_fr_trans/en-fr.csv"

    print("Start loading the entire dataset")
    entire = pd.read_csv(data_dir, nrows=1e6).dropna()
    print("Finish loading the entire dataset")

    save_dir = "/mnt/c/Users/dzhon/Desktop/myProjects/data/en_fr_trans"

    split(data=entire, save_dir=save_dir)
    print("Finish spliting the dataset")

    # Test the result of spliting
    try:
        train=pd.read_csv(save_dir+"/train.csv", nrows=5)
    except:
        print(f"Fail to load the training dataset from {save_dir}")

    tokenizer_en = spacy.load("en_core_web_sm")
    tokenizer_fr = spacy.load("fr_core_news_sm")

    print("Start building the vocab")
    build_source_target_vocab(raw=entire, source_tokenizer=tokenizer_en, 
                              target_tokenizer=tokenizer_fr, save_dir=save_dir)
    
    print("Finish building the vocab")

    # Test the result of building vocab
    try:
        source, target = load_vocab(load_dir=save_dir)
    except:
        print(f"Fail to load the vocab from {save_dir}")

def debug():
    data = pd.read_csv("/mnt/c/Users/dzhon/Desktop/myProjects/data/en_fr_trans/en-fr.csv", nrows=5)

    train, test = train_test_split(data, train_size=0.7)

    #test.reset_index(drop=True).to_csv(save_dir+"/test.csv", index=False)
    del test

    train, val = train_test_split(train, train_size=0.7)

    columns = data.columns
    a, b = columns
    tok = yield_tokens(data_iter=data[a], tokenizer=spacy.load("en_core_web_sm"))

    print(list(tok))

    build_source_target_vocab(data, source_tokenizer=spacy.load("en_core_web_sm"), target_tokenizer=spacy.load("fr_core_news_sm"), save_dir="")


if __name__ == "__main__":
    preprocess()





