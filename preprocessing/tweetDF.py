import pandas as pd
import os
import numpy as np
from data import train_positive_location, train_negative_location, \
  train_positive_sample_location, train_negative_sample_location
import pickle

def get_tweet_df():
    df_pos = pd.read_table(train_positive_sample_location, names=('tweet', 'id'))
    df_neg = pd.read_table(train_negative_sample_location, names=('tweet', 'id'))
    df_pos_1 = pd.DataFrame({
        'text': df_pos['tweet'],
        'sent': 0
    })

    df_neg_1 = pd.DataFrame({
        'text': df_neg['tweet'],
        'sent': 1
    })
    df_data = pd.concat([df_pos_1, df_neg_1], ignore_index=True, sort=False)
    abs_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(abs_path,"../data/tweetDF.pkl"), 'wb') as f:
        pickle.dump(df_data,f, pickle.HIGHEST_PROTOCOL)

def load_tweetDF():
    abs_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(abs_path,"../data/tweetDF.pkl"), 'rb') as f:
        tweetDF = pickle.load(f)
    return tweetDF

