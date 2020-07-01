import pandas as pd
import os
import numpy as np
from data import train_positive_location, train_negative_location
from data import replaced_train_full_positive_location, replaced_train_full_negative_location
from data import test_location
import pickle
from sklearn.utils import shuffle

def get_tweet_df(inputfiles, random_percentage):
    df_pos = pd.read_table(inputfiles[0], names=('tweet', 'id'))
    df_neg = pd.read_table(inputfiles[1], names=('tweet', 'id'))
    df_pos_1 = pd.DataFrame({
        'text': df_pos['tweet'],
        'sent': 0
    })
    df_pos_1_s = df_pos_1.sample(n=int(df_pos_1.shape[0]*random_percentage))

    df_neg_1 = pd.DataFrame({
        'text': df_neg['tweet'],
        'sent': 1
    })
    df_neg_1_s = df_neg_1.sample(n=int(df_neg_1.shape[0]*random_percentage))
    df_data = pd.concat([df_pos_1_s, df_neg_1_s], ignore_index=True, sort=False)
    df_data = shuffle(df_data)
    df_data = df_data[:50]
    abs_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(abs_path,"../data/tweetDF.pkl"), 'wb') as f:
        pickle.dump(df_data,f, pickle.HIGHEST_PROTOCOL)

    return df_data

def get_tweet_df_pred(input_test_loc):
    df_pred = pd.read_table(input_test_loc,names=('tweet', 'id'))
    df_pred = pd.DataFrame({
        'text':df_pred['tweet']
    })
    abs_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(abs_path, "../data/predDF.pkl"), 'wb') as f:
        pickle.dump(df_pred, f, pickle.HIGHEST_PROTOCOL)
    return df_pred


def load_tweetDF():
    abs_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(abs_path,"../data/tweetDF.pkl"), 'rb') as f:
        tweetDF = pickle.load(f)
    return tweetDF

def load_predDF():
    abs_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(abs_path,"../data/predDF.pkl"), 'rb') as f:
        predDF = pickle.load(f)
    return predDF


#output = get_tweet_df([train_positive_location, train_negative_location], random_percentage=0.3)
#print(output)
#get_tweet_df_pred(input_test_loc=test_location)
#output = load_predDF()
#print(output)
