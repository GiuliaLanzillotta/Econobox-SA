""" Base classifier definition.
    All other classifiers should implement this class.
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from abc import abstractmethod
from classifier import predictions_folder
import numpy as np
import os

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

    @staticmethod
    def score_model(true_classes, predicted_classes,
                    predicted_probabilities):
        """Calculates various scores associated with the model predictions.
            To use when testing the model."""
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(true_classes, predicted_classes)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(true_classes, predicted_classes)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(true_classes, predicted_classes)
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(true_classes, predicted_classes)
        print('F1 score: %f' % f1)
        # kappa
        kappa = cohen_kappa_score(true_classes, predicted_classes)
        print('Cohens kappa: %f' % kappa)
        # ROC AUC
        if not predicted_probabilities is None:
            auc = roc_auc_score(to_categorical(true_classes), predicted_probabilities)
            print('ROC AUC: %f' % auc)
        # confusion matrix
        matrix = confusion_matrix(true_classes, predicted_classes)
        print("Confusion matrix: ")
        print(matrix)


    def save_predictions(self, predictions_array):
        """
        Saves the predictions in the desired format
        :param predictions_array: (numpy array)
        :return: None
        """
        print("Saving predictions")
        #todo: fix the hashtag in the header
        abs_path = os.path.abspath(os.path.dirname(__file__))
        path = predictions_folder + self.name + "_predictions.csv"
        to_save_format = np.dstack((np.arange(1, predictions_array.size + 1), predictions_array))[0]
        np.savetxt(os.path.join(abs_path,path), to_save_format, "%d,%d",
                   delimiter=",", header="Id,Prediction", comments='')


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
