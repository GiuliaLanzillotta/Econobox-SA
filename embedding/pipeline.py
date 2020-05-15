# Offers the embedding pipeline methods
from embedding import embedding_dim, matrix_train_location
from embedding.glove import GloVeEmbedding
from embedding import sentence_embedding
from preprocessing.tokenizer import load_vocab
from data import sample_dimension, \
    train_negative_sample_location, train_positive_sample_location, test_location
import numpy as np


def build_training_matrix(label,
                          embedding,
                          input_files=None,
                          label_values=None,
                          aggregation_fun=sentence_embedding.sum_embeddings,
                          input_entries=sample_dimension,
                          sentence_dimesion = embedding_dim,
                          output_location = matrix_train_location):
    """
    Builds a matrix that associates each tweet to its embedding representation.
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
    # INITIALIZE ----------
    if label:
        out_dim1 = sentence_dimesion + 1
    else: out_dim1 = sentence_dimesion
    output = np.zeros((input_entries, out_dim1))

    # PROCESS THE FILES ----------
    counter = 0
    for i, file in enumerate(input_files):
        label_value = label_values[i]
        with open(file) as f:
            print("Working on ", file)
            # look at each line (=tweet)
            for l, line in enumerate(f):
                # Get the tweet embedding from an helper function
                # Note: you can safely ignore the max_len parameter
                # if you're not using the no-embedding aggregation function
                sentence_emb = aggregation_fun(line,embedding,
                                               max_len=sentence_dimesion).reshape(1, -1)
                                                            # we reshape to make sure it is a row vector
                # Save the tweet in the output matrix
                #TODO: make sure the order doesn't matter
                if not label:output[counter, :] = sentence_emb
                else:output[counter, :] = np.column_stack((sentence_emb, label_value))
                if l % 10000 == 0:
                    print(l)
                counter += 1
    # Save the output
    np.savez(output_location, output)
    return output

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

def run_embedding_pipeline(no_embedding=False,
                           output_location=matrix_train_location,
                           prediction_mode=False,
                           glove=True):
    """
    :param prediction_mode : (bool) whether the system is in prediction mode (i.e. loading
            the test data)
    Note: the glove parameter is not used now, because no other
    embedding is implemented. When a new embedding will be at disposal
    this parameter can be used to switch btw the two."""
    # Get the embedding
    print("Embedding pipeline")
    glove = get_glove_embedding(load_from_file=True,
                                load_Stanford=False,
                                file_name="glove+stanford.npz",
                                train=False,
                                save=False)
    embedding_function = None # default parameter will be used
    max_len = None # //          //
    if no_embedding:
        embedding_function = sentence_embedding.no_embeddings
        max_len = 50

    input_files = None
    if prediction_mode: input_files = [test_location]

    build_training_matrix(label=not prediction_mode,
                          embedding=glove,
                          aggregation_fun=embedding_function,
                          sentence_dimesion=max_len,
                          output_location=output_location,
                          input_files=input_files)

    if no_embedding: print("Number of sentence cut-offs: ",sentence_embedding.count)


# Until now:
# got 15 sentences cut-off with a max_len of 50





