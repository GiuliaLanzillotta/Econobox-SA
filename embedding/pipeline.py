# Offers the embedding pipeline methods
from embedding import embedding_dim, matrix_train_location, matrix_test_location
from embedding.glove import GloVeEmbedding
from embedding.negative_sampling import NegSamplingEmbedding
from embedding import sentence_embedding
from preprocessing.tokenizer import load_vocab
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import TweetTokenizer
from data import sample_dimension, \
    train_negative_sample_location, train_positive_sample_location, test_location, test_dimension
import tensorflow as tf
import numpy as np
import random
import os

def load_file_chunk(start, chunksize, file):
    """
    Loads -chunksize- tweets from the specified file, starting from
    -start- tweet (excluded).
    """
    f = open(file, encoding='utf8')
    lines = f.readlines()
    selected = lines[int(start):int(start + chunksize)]
    return selected

def generate_training_matrix(embedding,
                             input_files=None,
                             label_values=None,
                             chunksize=100,
                             validation_split=0.2,
                             categorical=True,
                             aggregation_fun=sentence_embedding.sum_embeddings,
                             input_entries=sample_dimension,
                             sentence_dimesion = embedding_dim):
    """
    Generator version of the function "build training matrix".
    A generator is a function that builds the data at runtime.
    To use in case of big files.
    :param categorical: whether to turn the labels to categorical variables.
        :type categorical: bool
    :param validation_split: fraction of the input to use as validation.
        :type validation_split: float
    :param chunksize: dimension of the chunk to load
        :type chunksize: int
    :param input_entries: total size (in number of lines) of the input
    :param embedding: EmbeddingBase. The embedding class to use.
    :param input_files: list(str). List of input files to transform.
    :param label_values: list(int). List of label values corresponding to each input file.
    :param aggregation_fun: function. Aggregation function to use to get a sentence embedding.
    :param sentence_dimesion: specifies the dimensionality of a sentence. Note that
    this value is correlated with the aggregation function. For instance, the sum_embeddings
    aggregation function will require the same dimensionality of the embeddings.
    :return: a tuple (x, y)
    """
    print("in generate train matrix")
    if label_values is None:
        label_values = [0, 1]
    if input_files is None:
        input_files = [train_negative_sample_location,
                       train_positive_sample_location]
    assert len(label_values) == len(input_files), \
        "The number of files must equal the number of labels"

    # resolving paths
    abs_path = os.path.abspath(os.path.dirname(__file__))
    input_paths = [os.path.join(abs_path,f) for f in input_files]

    validation_size = int(validation_split*input_entries//len(input_files))
    start = 0
    while start < int(input_entries//len(input_files) - validation_size):
        # Here we load a chunk from the positive file
        # and a chunk from the negative file
        sentences = []
        for file in input_paths:
            sentences += load_file_chunk(start, chunksize, file)
        labels = []
        for lv in label_values: labels+=[lv]*chunksize
        # embedding the sentence
        sentences_emb = [aggregation_fun(line, embedding,
                                         max_len=sentence_dimesion).reshape(1, -1) for line in sentences]
        # we reshape to make sure it is a row vector
        # shuffling
        temp = list(zip(sentences_emb, labels))
        random.shuffle(temp)
        sentences, labels = zip(*temp)
        sentences = np.array(sentences).reshape(chunksize*2,-1)
        if categorical: labels = tf.keras.utils.to_categorical(labels)
        start += chunksize
        yield (sentences, labels)


def get_validation_data(embedding,
                        input_files=None,
                        label_values=None,
                        validation_split=0.2,
                        categorical=True,
                        aggregation_fun=sentence_embedding.sum_embeddings,
                        input_entries=sample_dimension,
                        sentence_dimesion = embedding_dim):
    """
    Loads the whole validation set into memory.
    To use together with the generator function in case of big files.
    :param validation_split: fraction of the input to use as validation.
        :type validation_split: float
    :param input_entries: total size (in number of lines) of the input
    :param categorical: whether to turn the labels to categorical variables.
        :type categorical: bool
    :param embedding: EmbeddingBase. The embedding class to use.
    :param input_files: list(str). List of input files to transform.
    :param label_values: list(int). List of label values corresponding to each input file.
    :param aggregation_fun: function. Aggregation function to use to get a sentence embedding.
    :param sentence_dimesion: specifies the dimensionality of a sentence. Note that
    this value is correlated with the aggregation function. For instance, the sum_embeddings
    aggregation function will require the same dimensionality of the embeddings.
    :return: a tuple (x, y)
    """
    print("Preparing validation data.")
    if label_values is None:
        label_values = [0, 1]
    if input_files is None:
        input_files = [train_negative_sample_location,
                       train_positive_sample_location]
    assert len(label_values) == len(input_files), \
        "The number of files must equal the number of labels"

    # resolving paths
    abs_path = os.path.abspath(os.path.dirname(__file__))
    input_paths = [os.path.join(abs_path,f) for f in input_files]

    validation_size = int(validation_split*input_entries//len(input_files))
    start =  input_entries//len(input_files) - validation_size
    # Here we load a chunk from the positive file
    # and a chunk from the negative file
    sentences = []
    for file in input_paths:
        sentences += load_file_chunk(start, validation_size, file)
    labels = []
    for lv in label_values: labels+=[lv]*validation_size
    # embedding the sentence
    sentences_emb = [aggregation_fun(line, embedding,
                                     max_len=sentence_dimesion).reshape(1, -1) for line in sentences]
    # we reshape to make sure it is a row vector
    # shuffling
    temp = list(zip(sentences_emb, labels))
    random.shuffle(temp)
    sentences, labels = zip(*temp)
    sentences = np.array(sentences).reshape(validation_size*2, -1)
    if categorical: labels = tf.keras.utils.to_categorical(labels)
    print("Validation data prepared.")
    return (sentences, labels)


def build_training_matrix(label,
                          embedding,
                          input_files=None,
                          label_values=None,
                          aggregation_fun=sentence_embedding.sum_embeddings,
                          input_entries=sample_dimension,
                          use_tf_idf=False,
                          sentence_dimesion = 200,
                          output_location = matrix_train_location):
    """
    Builds a matrix that associates each tweet to its embedding representation.
    :param use_tf_idf: whether to use tf_idf weighting in the embedding
    :param input_entries: total size (in number of lines) of the input
    :param label: bool. Whether to insert a label column into the matrix.
    :param embedding: EmbeddingBase. The embedding class to use.
    :param input_files: list(str). List of input files to transform.
    :param label_values: list(int). List of label values corresponding to each input file.
    :param aggregation_fun: function. Aggregation function to use to get a sentence embedding.
    :param sentence_dimesion: specifies the dimensionality of a sentence. Note that
    this value is correlated with the aggregation function. For instance, the sum_embeddings
    aggregation function will require the same dimensionality of the embeddings.
    :param output_location: str/path. Where to save the output matri.
    :return: np.ndarray. The filled matrix.
    """

    if label_values is None:
        label_values = [0, 1]
    if input_files is None:
        input_files = [train_negative_sample_location,
                       train_positive_sample_location]

    if label: out_dim1 = sentence_dimesion + 1
    else: out_dim1 = sentence_dimesion
    output = np.zeros((input_entries, out_dim1))
    # PROCESS THE FILES ----------
    counter = 0
    if use_tf_idf:
        # Preparing the TF-IDF vectorizer
        vocabulary = embedding.vocabulary
        tokenizer = TweetTokenizer()
        vectorizer = TfidfVectorizer(vocabulary=vocabulary,
                                     tokenizer=tokenizer.tokenize)
    for i, file in enumerate(input_files):
        label_value = label_values[i]
        with open(file, encoding="utf8") as f:
            print("Working on ", file)
            # INITIALIZE ----------
            if use_tf_idf:
                with open(file, encoding="utf8") as f2:
                    print("TF-IDF vectorization of the file.")
                    tf_idf = vectorizer.fit_transform(f2)
            # look at each line (=tweet)
            for l, line in enumerate(f):
                # Get the tweet embedding from an helper function
                # Note: you can safely ignore the max_len parameter
                # if you're not using the no-embedding aggregation function

                if not label: line = line[3:] #cutting the first 3 characters if we're making
                # predictions since the test data has enumerated lines.

                #getting the tfidf weights if requested
                weights = None
                if use_tf_idf: weights=tf_idf[l,:].data
                sentence_emb = aggregation_fun(line,embedding,weights=weights)
                # we reshape to make sure it is a row vector
                # Save the tweet in the output matrix
                if not label:output[counter, :] = sentence_emb
                else: output[counter, :] = np.append(sentence_emb, np.array(label_value))
                if l % 10000 == 0:
                    print(l)
                counter += 1
    print("Number of lines read:",counter)
    print("Saving ", output_location)
    np.savez(output_location, output)

def get_glove_embedding(vocabulary_file="vocab.pkl",
                        cooc_file="cooc.pkl",
                        load_from_file=False,
                        file_name = None,
                        load_Stanford=False,
                        train=False,
                        save=False,
                        train_epochs=10,
                        train_eta=1e-3):
    """
    Creates a GloveEmbedding class.
    By default it will leave the embedding matrix randomly initialized.
    :param vocabulary_file: the name of the vocavulary in use
    :param cooc_file: (str) the name of the cooc matrix to use
    :param load_from_file: whether to load the embedding from file
    :param file_name: file from which to load the embedding
    :param load_Stanford: whether to load the Stanford pre-trained embedding
    :param train: whether to train the embedding
    :param save: whether to save the embedding to file
    :param train_epochs: number of training epochs
    :param train_eta: training learning rate
    :return:
    """
    print("In get glove embedding")
    vocab = load_vocab(vocabulary_file)
    if file_name is None: file_name = "glove_emb.npz"
    gloVe_embedding = GloVeEmbedding(file_name,
                                     embedding_dim,
                                     vocab,
                                     cooc_file,
                                     load=load_from_file)
    if load_Stanford: gloVe_embedding.load_stanford_embedding()
    if train: gloVe_embedding.train_embedding(epochs=train_epochs,
                                              eta=train_eta)
    if save: gloVe_embedding.save_embedding()
    return gloVe_embedding

embedding_funcs = {
    "no_embedding":sentence_embedding.no_embeddings,
    "sum_embeddings":sentence_embedding.sum_embeddings
    #"transformer_emb":sentence_embedding.embedize
}

def get_neg_sampling_embedding(vocabulary_file="vocab.pkl",
                        cooc_file="cooc.pkl",
                        neg_cooc_file="neg_cooc.pkl",
                        load_from_file=False,
                        file_name = None,
                        train=False,
                        save=False,
                        train_epochs=10,
                        train_eta=1e-3):
    """
    Creates a NegSampling class.
    By default it will leave the embedding matrix randomly initialized.
    :param vocabulary_file: the name of the vocavulary in use
    :param cooc_file: (str) the name of the cooc matrix to use
    :param neg_cooc_file: (str) the name of the neg_cooc matrix to use    
    :param load_from_file: whether to load the embedding from file
    :param file_name: file from which to load the embedding
    :param train: whether to train the embedding
    :param save: whether to save the embedding to file
    :param train_epochs: number of training epochs
    :param train_eta: training learning rate
    :return:
    """
    print("In get neg sampling embedding")
    vocab = load_vocab(vocabulary_file)
    if file_name is None: file_name = "neg_sampling_emb.npz"
    neg_sampling_emb = NegSamplingEmbedding(file_name,
                                     embedding_dim,
                                     vocab,
                                     cooc_file,
                                     neg_cooc_file,   
                                     load=load_from_file)
    if train: neg_sampling_emb.train_embedding(epochs=train_epochs,
                                              eta=train_eta)
    if save: neg_sampling_emb.save_embedding()
    return neg_sampling_emb


def run_embedding_pipeline(embedding_fun,
                           input_files=None,
                           input_labels=None,
                           input_entries=sample_dimension,
                           output_location=matrix_test_location,
                           prediction_mode=True,
                           use_tf_idf=True,
                           glove=True,
                           **kwargs):
    """
    :param input_labels: (list of ints) list of input labels
    :param embedding_fun: (str) the name of the embedding function to use.
        Only supported values at the moment:
            - "no_embedding"
            - "sum_embeddings"
            - "transformer_emb"
    :param input_files: (list(str)) relative paths to the input files
    :param glove: (bool) whether to use Glove embedding (for now the only available one)
    :param output_location: (str) where to save the output matrix
    :param input_entries: (int) dimension of the input file
    :param prediction_mode : (bool) whether the system is in prediction mode (i.e. loading
            the test data)
    Note: the glove parameter is not used now, because no other
    embedding is implemented. When a new embedding will be at disposal
    this parameter can be used to switch btw the two."""
    # Get the embedding
    if input_labels is None:
        input_labels = [0, 1]
    print("Embedding pipeline")
    max_len = kwargs.get("max_len",200)
    embedding_function = embedding_funcs[embedding_fun]
    ## 1. extracting the embedding to use
    print(embedding_fun)

    if embedding_fun!="transformer_emb":
        if glove:
            embedding = get_glove_embedding(vocabulary_file="vocab.pkl",
                                        load_from_file=True,
                                        load_Stanford=False,
                                        file_name="glove+stanford.npz",
                                        train=False,
                                        save=True)
        else:
            embedding = get_neg_sampling_embedding(vocabulary_file="vocab.pkl",
                        cooc_file="cooc.pkl",
                        neg_cooc_file="neg_cooc.pkl",
                        load_from_file=False,
                        file_name = None,
                        train=False,
                        save=False)
    else: embedding=kwargs.get("embedding", "roberta-base")
    ## 2. resolving paths
    if prediction_mode:
        input_files = [test_location]
        input_entries = test_dimension
    abs_path = os.path.abspath(os.path.dirname(__file__))
    output_path = os.path.join(abs_path,output_location)
    input_paths = [os.path.join(abs_path,f) for f in input_files]

    

    build_training_matrix(label=not prediction_mode,
                          label_values=input_labels,
                          embedding=embedding,
                          use_tf_idf=use_tf_idf,
                          aggregation_fun=embedding_function,
                          sentence_dimesion=max_len,
                          output_location=output_path,
                          input_entries=input_entries,
                          input_files=input_paths)

    if embedding_fun=="no_embedding":
        print("Number of sentence cut-offs: ",sentence_embedding.count)
        print("Average frequency of in-vocabulary words: ", sentence_embedding.frequency/input_entries)


# Until now:
# got 15 sentences cut-off with a max_len of 50
#
# 103 cut-offs with a max len of 50 in full text file






