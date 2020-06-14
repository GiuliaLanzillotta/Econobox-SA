# Here we implement all functions/classes relative to tokenizing 
from data import train_positive_location, train_negative_location, \
  train_positive_sample_location, train_negative_sample_location
from embedding import stanford_embedding_location
from preprocessing import vocabularies_folder
from nltk.tokenize.casual import TweetTokenizer
from collections import Counter
import os
import pickle
import re

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

# todo incorporate standardization in tokenization

def build_stanford_vocab(file_name="stanford_vocab.pkl"):
    """ Extracts the vocabulary from the Stanford embedding and saves it
        as a new vocabulary. """
    abs_path = os.path.abspath(os.path.dirname(__file__))
    f = open(os.path.join(abs_path, stanford_embedding_location), encoding='utf8')
    words = []
    for line in f:
        values = line.split()
        word = values[0]
        words.append(word)
    f.close()
    # counting the words
    counter = Counter(words)
    words_count = dict(counter)
    # building voabulary
    vocab = {k:i for i,k in enumerate(words_count)}
    # saving the vocabulary
    with open(os.path.join(abs_path,vocabularies_folder+file_name), 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    return vocab

def build_vocab(frequency_treshold=10,
                file_name="vocab.pkl",
                use_base_vocabulary=False,
                base_vocabulary_name="stanford_vocab.pkl",
                input_files=None):
    """
    Builds a vocabulary from the 2 training files. 
    :param frequency_treshold: int
        Treshold that will be used to filter the vocabulary. 
        All the words that appear with a frequency lower or equal
        to the treshold will be deleted from the vocabulary.
    :param file_name: str
        Name of the file to which the vocabulary will be saved.
    :param use_base_vocabulary: whether to load a second vocabulary to
        filter the words. This second vocabulary will act as a second filter:
        all the words that are not in this second vocabulary will be excluded
        from the first as well.
    :param base_vocabulary_name
        :type base_vocabulary_name str
    :param input_files: list(str)
        Name of the files from which to build the vocabulary
    :returns dict
        The vocabulary.
    """

    if input_files is None:
        input_files = [train_positive_sample_location, train_negative_sample_location]

    if use_base_vocabulary:
        base_vocabulary = load_vocab(base_vocabulary_name)
        base_vocabs = base_vocabulary.keys()

    abs_path = os.path.abspath(os.path.dirname(__file__))
    words = []
    for f in input_files:
        print("Reading ",f)
        raw = open(os.path.join(abs_path, f),  "r", encoding="utf8").read()
        more_words = tokenize_text(raw)
        words.extend(more_words)
    # counting the words
    counter = Counter(words)
    words_count = dict(counter)
    # filtering
    def filter_in(word, frequency):
        value = frequency >= frequency_treshold
        if use_base_vocabulary: value = value and word in base_vocabs
        return value
    filtered_words = [k for k, v in words_count.items() if filter_in(k,v)]
    # building voabulary 
    vocab = {k:i for i,k in enumerate(filtered_words)}
    # saving the vocabulary 
    with open(os.path.join(abs_path,vocabularies_folder+file_name), 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    return vocab


def load_vocab(file_name):
    """
    Loads the vocabulary at the given location.
    """
    abs_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(abs_path,vocabularies_folder+file_name), 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def load_inverse_vocab(file_name):
    """
        Loads the idx2word vocabulary at the given location.
        """
    vocab = load_vocab(file_name)
    idx2word = {item[1]:item[0] for item in vocab.items()}
    return idx2word

def get_vocab_dimension(file_name):
    """
    Opens the specified vocabulary to count the number of keys.
    """
    vocab = load_vocab(file_name)
    return len(vocab.keys())
