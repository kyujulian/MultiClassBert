import os
import numpy as np
import pandas as pd
import tensorflow as tf
from data_handle import get_csv_data, normalize_tweets,split_dataset

from model import build_model, train_model, transformer_layer, tokenizer, fast_tokenizer


from utils import *
from constants import *

from data_handle import *

def main():
    #read data
    data = get_csv_data(DATA_DIR, "twitter_corpus.csv")
    normalize_tweets(data)
    train_dataset, dev_dataset, test_dataset = get_datasets(data,0.8, fast_tokenizer)
    model = build_model(transformer_layer)
    train_model(train_dataset, dev_dataset, model)

    model.save("./model_saved/classifier_hf.h5")





