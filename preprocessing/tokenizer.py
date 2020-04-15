# Here we implement all functions/classes relative to tokenizing 
from preprocessing import train_positive_location, train_negative_location, \
  train_positive_sample_location, train_negative_sample_location, vocab_location
from nltk.tokenize.casual import TweetTokenizer
from collections import Counter
import os
import pickle

def tokenize_text(text):
    """
    Transforms the specified files in tokens using the Twitter tokenizer.
    @params: str
        Input text to tokenize
    @returns: list(str)
        Returns the tokens as a list of strings.
    """
    tokenizer = TweetTokenizer()
    # tokenizing the text
    tokens = tokenizer.tokenize(text)
    words = [w.lower() for w in tokens]
    return words

def build_vocab(frequency_treshold, file_name = vocab_location):
    """
    Builds a vocabulary from the 2 training files. 
    @param frequency_treshold: int
        Treshold that will be used to filter the vocabulary. 
        All the words that appear with a frequency lower or equal
        to the treshold will be deleted from the vocabulary. 
    @param file_name: str
        Name of the file to which the vocabulary will be saved.
    @returns dict
        The vocabulary.
    """
    abs_path = os.path.abspath(os.path.dirname(__file__))
    words = []
    for f in [train_positive_sample_location,train_negative_sample_location]:
        print("Reading ",f)
        raw = open(os.path.join(abs_path, f),  "r").read()
        more_words = tokenize_text(raw)
        words.extend(more_words)
    # counting the words
    counter = Counter(words)
    words_count = dict(counter)
    # filtering 
    filtered_words = [k for k, v in words_count.items() if v >= frequency_treshold]
    # building voabulary 
    vocab = {k:i for i,k in enumerate(filtered_words)}
    # saving the vocabulary 
    with open(file_name, 'wb') as f: 
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    return vocab

def load_vocab(file_name = vocab_location):
    """
    Loads the vocabulary at the given location.
    """
    with open(file_name, 'rb') as f:
        vocab = pickle.load(f)
    return vocab
