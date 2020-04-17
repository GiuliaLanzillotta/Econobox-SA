# Defines the base methods to be implemented by any Embedding
from abc import abstractmethod

import numpy as np
import pickle
import os

class EmbeddingBase(object):
    """
    Base Embedding class.
    This class defines the base methods that should be implemented
    by any embedding.
    """
    def __init__(self, embedding_location,
                 embedding_dimension,
                 vocabulary,
                 cooc,
                 load = False):
        """

        :param embedding_location: the file where the embedding will be stored
        :param embedding_dimension: pre-defined dimension for the embedding space
        :param vocabulary: (dict) the vocabulary to use fot the embedding
        :param cooc: (str) the name of the cooc matrix to use for the embedding
        :param load: whether to load the matrix from file using the load method.
                If False, the matrix will be randomly initialized.
        """
        super(EmbeddingBase, self).__init__()
        self.embedding_location = embedding_location
        self.embedding_dimension = embedding_dimension
        self.vocabulary_dimension = len(vocabulary.keys())
        self.vocabulary = vocabulary
        self.cooc = cooc
        if load: self.embedding_matrix = self.load_embedding()
        else: self.embedding_matrix =  np.random.normal(size=(self.vocabulary_dimension,
                                                        embedding_dimension))


    @abstractmethod
    def train_embedding(self, **args):
        """
        Trains the embedding by producing an embedding matrix
        and saves the matrix to file.
        @:return the embedding matrix
        """
        pass

    def load_embedding(self, location=None):
        """
        Loads the embedding matrix from file into the class
        :param location: alternative location to load the embedding from
        :return: the embedding matrix if present
        """
        abs_path = os.path.abspath(os.path.dirname(__file__))
        if location is None: location = self.embedding_location
        self.embedding_matrix = np.load(os.path.join(abs_path,location))["arr_0"]
        return self.embedding_matrix

    def save_embedding(self, location=None):
        """
        Saves the embedding matrix to file
        :param location: alternative location to save the embedding to
        :return: None
        """
        abs_path = os.path.abspath(os.path.dirname(__file__))
        if location is None: location = self.embedding_location
        np.savez(os.path.join(abs_path,location), self.embedding_matrix)