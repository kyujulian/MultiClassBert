import pandas as pd
import numpy as np
import tensorflow as tf
from constants import *
from utils import fast_encode
import os

ONE_HOT_DICT = {
    "negative" : 0,
    "positive" : 1,
    "irrelevant" : 2,
    "neutral" :3,
}


ONE_HOT_REVERSE_DICT = {v: k for k, v in ONE_HOT_DICT.items()}

def get_csv_data(data_dir : str, filename) :
    corpus = pd.read_csv(os.path.join(data_dir, filename))
    return corpus


#Specific to this dataset
def normalize_tweets(raw_data):
    """
    Como os dados são bem "skewed" no sentido de haverem desproporcionalmente mais classes "neutras" e "irrelevantes" 
    do que as demais. Esse processo ajuda na performance do modelo.
    """

    neutral_tweets = raw_data[raw_data.Sentiment == 'neutral']

    tweets_without_neutral = raw_data[raw_data.Sentiment != 'neutral']
    neutral_tweets_600 = neutral_tweets.iloc[:600]
    scaled_tweets = pd.concat([neutral_tweets_600, tweets_without_neutral])

    tweets_without_irrelevant = scaled_tweets[scaled_tweets.Sentiment != 'irrelevant']
    irrelevant_tweets = scaled_tweets[scaled_tweets.Sentiment == 'irrelevant'].iloc[:600]
    scaled_tweets = pd.concat([tweets_without_irrelevant, irrelevant_tweets])

    scaled_tweets.Sentiment.value_counts().plot(kind='bar')

    scaled_tweets = scaled_tweets.reset_index(drop=True)


# def one_hot_mapping(data, oh_dict=ONE_HOT_DICT):
#     data.map(lambda x: oh_dict[x])


def split_dataset(data, train_ratio,fast_tokenizer , label_column="Sentiment", save_to_file=True):
    '''
    Função para dividir o dataset em treino, validação e teste.
    return:
        train_X, train_y, dev_X, dev_y, test_x,test_y
    '''

    test_ratio = (1 - train_ratio)/2
    n_of_samples = len(data.index)

    labels = tf.keras.utils.to_categorical(data[label_column].map(lambda x: ONE_HOT_DICT[x]), num_classes=4)

    train_idx = np.round(n_of_samples * train_ratio).astype(np.int32)

    val_idx = train_idx + np.round(n_of_samples * test_ratio).astype(np.int32)

    dataframe_labels = data[[label_column]]
    train = data[0:train_idx]
    dataframe_train_labels = dataframe_labels[0:train_idx]
    train_labels = labels[0: train_idx]

    dev = data[train_idx: val_idx]
    dataframe_dev_labels = dataframe_labels[train_idx: val_idx]
    dev_labels = labels[train_idx : val_idx]

    test = data[val_idx:]
    dataframe_test_labels = dataframe_labels[val_idx:]
    test_labels = labels[val_idx :]

    if save_to_file:
        pd.concat((train,dataframe_train_labels),axis=1).to_csv("./data/twitter_train.csv")
        pd.concat((dev,dataframe_dev_labels),axis=1).to_csv("./data/twitter_dev.csv")
        pd.concat((test,dataframe_test_labels),axis=1).to_csv("./data/twitter_test.csv")



    train = fast_encode(train.TweetText.astype(str), fast_tokenizer, maxlen=128)
    test = fast_encode(test.TweetText.astype(str), fast_tokenizer, maxlen=128)
    dev = fast_encode(dev.TweetText.astype(str), fast_tokenizer, maxlen=128)

    return (train, train_labels, dev, dev_labels, test, test_labels)


def make_datasets(x_train, y_train,
                  x_dev, y_dev,
                  x_test, y_test):

    n_steps = x_train.shape[0] // BATCH_SIZE
    train_dataset = (
    tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .shuffle(2048)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    dev_dataset = (
    tf.data.Dataset
        .from_tensor_slices((x_dev, y_dev))
        .shuffle(2048)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    test_dataset = (
    tf.data.Dataset
        .from_tensor_slices((x_test, y_test))
        .batch(BATCH_SIZE)
    )

    return train_dataset, dev_dataset, test_dataset, n_steps

def get_datasets(data,train_ratio,fast_tokenizer, label_column="Sentiment", save_to_file=True):

    train, train_labels, dev, dev_labels, test, test_labels = split_dataset(data,train_ratio,fast_tokenizer,label_column,save_to_file=save_to_file)

    return make_datasets(train, train_labels, dev, dev_labels, test, test_labels)