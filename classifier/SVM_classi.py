#Here we should implement a SVM classifier that thakes as input
#the embedding for a sentence and prints out the class of the tweet
from classifier.classifier_base import ClassifierBase
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.utils import _joblib
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from keras.utils import to_categorical
from classifier import models_store_path
import numpy as np
import pickle

class SVM_classi(ClassifierBase):
    """SVM classifier"""
    def __init__(self,
                 embedding_dimension,
                 name="oursvm"):
        super().__init__(embedding_dimension, name=name)
        self.history = None
        self.model = None

    def build(self,**kwargs):
        print("Building model")

        c = kwargs.get("c")
        kernel = kwargs.get("kernel")
        gamma = kwargs.get("gamma")
        max_iter = kwargs.get("max_iter")

        if not c: c = 1
        if not kernel: kernel = "rbf"
        if not gamma: gamma = "scale"
        if not max_iter: max_iter = -1

        self.model = SVC(C=c,kernel=kernel,gamma=gamma, max_iter=max_iter)


    def train(self, x, y, **kwargs):
        """
        Training the model and saving the history of training
        """
        print("Training model")
        self.history = self.model.fit(x,y)
        crosvalscore = cross_val_score(self.model, x, y, cv=5, scoring='f1')
        print(crosvalscore)

    def test(self,x,y, **kwargs):
        print("Testing model")
        y_pred = self.model.predict(x)
        score = f1_score(y, y_pred)
        return(score)

    def make_predictions(self, x, save=True, **kwargs):
        print("Making predictions")
        preds = self.model.predict(x)
        preds[preds == 0] = -1
        if save: self.save_predictions(preds)
        return preds

    def save(self, overwrite=True, **kwargs):
        print("Saving model")
        #path = models_store_path+self.name
        _joblib.dump(self.model, 'oursvm.pkl')

    def load(self, **kwargs):
        print("Loading model")
        #path = models_store_path+self.name
        self.model = _joblib.load('oursvm.pkl')









