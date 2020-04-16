""" Base classifier definition.
    All other classifiers should implement this class.
"""
from abc import abstractmethod


class ClassifierBase(object):

    def __init__(self,embedding_dimension,name):
        """
        :param embedding_dimension: this will be the input dimension
        :param name: the model will be saved as name+"_classifier"
        """
        self.input_dim = embedding_dimension
        self.name = name

    @abstractmethod
    def build(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self,
              x, y,
              *args,
              **kwargs):
        pass

    @abstractmethod
    def test(self,
             x, y,
             **kwargs):
        pass

    @abstractmethod
    def save(self, overwrite=True, **kwargs):
        pass

    @abstractmethod
    def load(self, **kwargs):
        """
        Loads the model from file.
        """
        pass
