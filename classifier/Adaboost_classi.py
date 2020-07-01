from classifier.classifier_base import ClassifierBase
from classifier import models_store_path
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import _joblib
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class Adaboost_classi(ClassifierBase):
    """Random Forest classifier"""

    def __init__(self,
                 embedding_dimension,
                 name="ourADA"):
        super().__init__(embedding_dimension, name=name)
        self.history = None
        self.model = None

    def build(self, **kwargs):
        print("Building model.")

        base_estimator = kwargs.get("base_estimator")
        n_estimators = kwargs.get("n_estimators")
        learning_rate = kwargs.get("learning_rate")
        algorithm = kwargs.get("algorithm")


        if not base_estimator: base_estimator = None
        if not n_estimators: n_estimators = 50
        if not learning_rate: learning_rate = 1
        if not algorithm: algorithm = "SAMME.R"


        self.model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate,
                                            algorithm=algorithm)

    def train(self, x, y, **kwargs):
        """
        Training the model and saving the history of training
        """
        print("Training model")
        self.history = self.model.fit(x,y)
        crosvalscore = cross_val_score(self.model, x, y, cv=5)
        print(crosvalscore)

    def test(self, x,y, **kwargs):
        print("Testing model")
        y_pred = self.model.predict(x)
        print(y_pred)
        score = accuracy_score(y, y_pred, normalize=False)
        return(score)

    def make_predictions(self, x, save=True, **kwargs):
        print("Making predictions")
        preds = self.model.predict(x)
        preds[preds == 0] = -1
        if save: self.save_predictions(preds)

    def save(self, overwrite=True, **kwargs):
        print("Saving model")
        path = models_store_path+self.name
        pickle.dump(self, open(path, 'wb'))
        _joblib.dump(self.model, 'ourADA.pkl')

    def load(self, **kwargs):
        print("Loading model")
        path = models_store_path+self.name
        self.model = _joblib.load('ourADA.pkl')