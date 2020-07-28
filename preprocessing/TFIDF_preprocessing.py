from preprocessing import tweetDF
import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from data import train_positive_sample_location, train_negative_sample_location
import pickle
import os

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



def TFIDF_VEC_train(input_files_train,random_percentage):
    print("in tfidf_vec_train")
    data_train = tweetDF.get_tweet_df(input_files=input_files_train, random_percentage=random_percentage)
    X_train= data_train.tweet.values
    y_train = data_train.label.values
    print("y_train",y_train)
    X_train_preprocessed = np.array([text_preprocessing(text) for text in X_train])
    print("done with text preprocessing")
    vectorizer = TfidfVectorizer(ngram_range=(1, 1),
                             binary=True,
                             smooth_idf=False)
    print("fit transofrm data")
    tfidf = vectorizer.fit(X_train_preprocessed)
    X_train_tfidf = vectorizer.transform(X_train_preprocessed)

    abs_path = os.path.abspath(os.path.dirname(__file__))
    pickle.dump(tfidf, open(os.path.join(abs_path, "../data/tfidf.pkl"), 'wb'))
    return [X_train_tfidf, y_train]

def TFIDF_VEC_pred(input_files):
    data_test = tweetDF.get_tweet_df_pred(input_files=input_files)
    X_test = data_test.tweet.values
    print(X_test)
    X_test_preprocessed = np.array([text_preprocessing(text) for text in X_test])

    abs_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(abs_path, "../data/tfidf.pkl"), 'rb') as f:
        vectorizer = pickle.load(f)
    X_test_tfidf = vectorizer.transform(X_test_preprocessed)
    print(X_test_tfidf)

    return X_test_tfidf



#data = TFIDF_VEC_train(input_files=[replaced_train_full_positive_location_d, replaced_train_full_negative_location_d], random_percentage=0.02)