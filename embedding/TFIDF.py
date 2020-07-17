from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import tweetDF
from data import train_positive_sample_location, train_negative_sample_location
import pickle
import os
import pandas as pd

def get_tweet_tf_idf_data(data):
    tf_idf = TfidfVectorizer(ngram_range=(1,3),
                             binary=True,
                             smooth_idf=True)

    data_tfidf_text = tf_idf.fit_transform(data.text.tolist())
    print(type(data_tfidf_text))
    #abs_path = os.path.abspath(os.path.dirname(__file__))
    #with open(os.path.join(abs_path,"../data/tf_idf_vectorized.pkl"), 'wb') as f:
    #    pickle.dump(data_tfidf,f, pickle.HIGHEST_PROTOCOL)

    return (data_tfidf_text)

def load_tweet_tf_idf_data():
    abs_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(abs_path,"../data/tf_idf_vectorized.pkl"), 'rb') as f:
        data_tfidf = pickle.load(f)
    return data_tfidf


#data = tweetDF.get_tweet_df(inputfiles=[train_positive_sample_location, train_negative_sample_location], random_percentage=1)
#get_tweet_tf_idf_data(data[:50])