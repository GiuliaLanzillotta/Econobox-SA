from preprocessing import tweetDF
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from data import train_positive_sample_location, train_negative_sample_location


def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    return s


def get_tfidf_train_data(input_files, random_percentage):
    data_df = tweetDF.get_tweet_df(input_files=input_files,
                                   random_percentage=random_percentage)

    data_df_preprocessed = np.array([text_preprocessing(text) for text in data_df.tweet.values])
    tf_idf = TfidfVectorizer(ngram_range=(1, 3),
                         binary=True,
                         smooth_idf=False)
    X_train_tfidf = tf_idf.fit_transform(data_df_preprocessed)
    return [X_train_tfidf, data_df.label.values]


def get_tfidf_test_data(input_files):
    data_pred_df = tweetDF.get_tweet_df_pred(input_files=input_files)
    data_df_pred_preprocessed = np.array([text_preprocessing(text) for text in data_pred_df.tweet.values])
    tf_idf = TfidfVectorizer(ngram_range=(1, 3),
                         binary=True,
                         smooth_idf=False)
    X_pred_tfidf = tf_idf.fit_transform(data_df_pred_preprocessed)
    return X_pred_tfidf





