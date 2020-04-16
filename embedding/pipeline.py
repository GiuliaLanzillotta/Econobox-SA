# Offers the embedding pipeline methods
from embedding import embedding_dim, glove_embedding_location, matrix_train_location
from embedding.glove import GloVeEmbedding
from embedding import sentence_embedding
from embedding.embedding_base import EmbeddingBase
from preprocessing.tokenizer import load_vocab, tokenize_text
from preprocessing import sample_dimension, train_negative_sample_location, train_positive_sample_location
import numpy as np

def build_embedding_matrix(label,
                           embedding,
                           input_files=None,
                           label_values=None,
                           aggregation_fun=sentence_embedding.sum_embeddings,
                           input_entries=sample_dimension,
                           output_location = matrix_train_location):
    """
    Builds a matrix that associates each tweet to its embedding representation.
    :param input_entries: total size (in number of lines) of the input
    :param label: bool. Whether to insert a label column into the matrix.
    :param embedding: EmbeddingBase. The embedding class to use.
    :param input_files: list(str). List of input files to transform.
    :param label_values: list(int). List of label values corresponding to each input file.
    :param aggregation_fun: function. Aggregation function to use to get a sentence embedding.
    :param output_location: str/path. Where to save the output matri.
    :return: np.ndarray. The filled matrix.
    """

    if label_values is None:
        label_values = [0, 1]
    if input_files is None:
        input_files = [train_negative_sample_location,
                       train_positive_sample_location]
    # INITIALIZE ----------
    if label: out_dim1 = embedding_dim + 1
    else: out_dim1 = embedding_dim
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
                sentence_emb = aggregation_fun(line,embedding).reshape(1, -1)  # making sure it is a row vector
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

def get_glove_embedding(vocabulary_file=None,
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
    :param vocabulary_file: the path to the vocavulary in use
    :param load_from_file: whether to load the embedding from file
    :param file_name: file from which to load the embedding
    :param load_Stanford: whether to load the Stanford pre-trained embedding
    :param train: whether to train the embedding
    :param save: whether to save the embedding to file
    :param train_epochs: number of training epochs
    :param train_eta: training learning rate
    :return:
    """
    if vocabulary_file is None: vocab = load_vocab()
    else: vocab = load_vocab(vocabulary_file)
    if file_name is None: file_name = glove_embedding_location
    gloVe_embedding = GloVeEmbedding(file_name,
                                     embedding_dim,
                                     vocab,
                                     load=load_from_file)
    if load_Stanford: gloVe_embedding.load_stanford_embedding()
    if train: gloVe_embedding.train_embedding(epochs=train_epochs,
                                              eta=train_eta)
    if save: gloVe_embedding.save_embedding()
    return gloVe_embedding

def run_embedding_pipeline(glove=True):
    # Get the embedding
    print("Embedding pipeline")
    glove = get_glove_embedding(load_from_file=True)
    build_embedding_matrix(label=True,
                           embedding=glove)






