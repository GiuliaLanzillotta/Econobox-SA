""" Base classifier definition.
    All other classifiers should implement this class.
"""
from abc import abstractmethod
from classifier import predictions_folder
import numpy as np

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

    def save_predictions(self, predictions_array):
        """
        Saves the predictions in the desired format
        :param predictions_array: (numpy array)
        :return: None
        """
        print("Saving predictions")
        path = predictions_folder + self.name + "_predictions.csv"
        to_save_format = np.dstack((np.arange(1, predictions_array.size + 1), predictions_array))[0]
        np.savetxt(path, to_save_format, "%d,%d",
                   delimiter=",", header="Id,Predictions")


    @abstractmethod
    def make_predictions(self, x, save=True, **kwargs):
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
