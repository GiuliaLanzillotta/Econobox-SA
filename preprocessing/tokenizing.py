# Here we implement all functions/classes relative to tokenizing 
from __init__.py import train_positive_location, train_negative_location
from nltk.tokenize.casual import TweetTokenizer
from collections import Counter
import pickle

def build_vocab(frequency_treshold, file_name = "vocab.pkl"):
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
    # reading the files
    raw_pos = open(train_positive_location,  "r").read()
    raw_neg = open(train_negative_location,  "r").read()
    # tokenizing the files
    tokenizer = TweetTokenizer()
    tokens_pos = tokenizer.tokenize(raw_pos)
    tokens_neg = tokenizer.tokenize(raw_neg)
    words = [w.lower() for w in tokens_pos]
    words_neg = [w.lower() for w in tokens_neg]
    words.extend(words_neg)
    # counting the words
    counter = Counter(words)
    words_count = dict(counter)
    # filtering 
    filtered_words = [k for k, v in words_count.items() if v >= frequency_treshold]
    # building voabulary 
    vocab = {i:k for i,k in enumerate(filtered_words)}
    # saving the vocabulary 
    with open(file_name, 'wb') as f: 
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    return vocab
