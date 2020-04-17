# All necessary tools to train a glove embedding
from abc import abstractmethod, ABC

from embedding import settings_location, glove_embedding_location, embedding_dim, \
    stanford_embedding_location
from embedding.embedding_base import EmbeddingBase
from preprocessing import sample_dimension, cooc_folder, vocabularies_folder
import numpy as np 
import pickle
import json
import os


class GloVeEmbedding(EmbeddingBase):
    """Implements GloVe version of Embedding"""

    def __init__(self,
                 embedding_location,
                 embedding_dimension,
                 vocabulary,
                 cooc,
                 input_dimension = sample_dimension,
                 load = False):
        super(GloVeEmbedding, self).__init__(embedding_location,
                                             embedding_dimension,
                                             vocabulary,
                                             cooc,
                                             load)
        abs_path = os.path.abspath(os.path.dirname(__file__))
        # read glove hyperparameters from settings
        print("Loading hyperparameters")
        with open(os.path.join(abs_path, settings_location), "r") as f:
            settings = json.load(f)
            glove_settings = settings["glove"]
        self.beta = glove_settings["beta"]
        self.alpha = glove_settings["alpha"]
        self.max = self.beta * input_dimension

    def train_embedding(self, epochs, eta):
        """
        Training GloVe embedding.
        This method uses SGD.
        :param epochs: number of epochs to train for
        :param eta: learning rate
        :return: the trained embedding matrix
        """
        # read co-occurrence matrix
        print("Opening co-occurrence matrix")
        with open(cooc_folder+self.cooc, 'rb') as f:
            cooc = pickle.load(f)
        # start glove training
        xs = self.embedding_matrix
        print("Started GloVe training")
        for epoch in range(epochs):
            print("epoch {}".format(epoch))
            for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
                logn = np.log(n)
                fn = min(1.0, (n / self.max) ** self.alpha)
                x, y = xs[ix, :], xs[jy, :]
                scale = 2 * eta * fn * (logn - np.dot(x, y))
                xs[ix, :] += scale * y
                xs[jy, :] += scale * x
        self.embedding_matrix = xs

    def load_stanford_embedding(self):
        """
        Loads Stanford GloVe embedding.
        NOTA BENE: Stanford embedding uses an embedding dimension of 200.
        :return: Stanford embedding
        """
        print("Loading pre-trained Stanford embedding")

        embeddings_index = {}
        abs_path = os.path.abspath(os.path.dirname(__file__))
        f = open(os.path.join(abs_path, stanford_embedding_location), encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        emb_matrix = np.zeros((self.vocabulary_dimension, self.embedding_dimension))
        for idx, word in enumerate(self.vocabulary.keys()):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                emb_matrix[idx + 1] = embedding_vector
            # Please note that the words for which there is no
            # embedding vector in the pre-trained embedding
            # will be left as 0 - which should be Neutral
        self.embedding_matrix = emb_matrix
        return emb_matrix

