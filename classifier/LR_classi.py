# Here we should implement a Logistic Regression classifier that
# takes as input the embedding for a sentence and prints as output the
# class of the tweet
from classifier.classifier_base import ClassifierBase
from classifier import models_store_path
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

class LR_classi(ClassifierBase):
    """Logistic Regression classifier"""

    def __init__(self,
                 embedding_dimension,
                 name="ourLR"):
        super().__init__(embedding_dimension, name=name)
        self.history = None
        self.model = None

    def build(self, **kwargs):
        print("Building model.")

        penalty = kwargs.get("penalty")
        c = kwargs.get("c")
        class_weight = kwargs.get("class_weight")
        solver = kwargs.get("solver")
        max_iter = kwargs.get("max_iter")

        if not penalty: penalty = "l2"
        if not c: c = 1
        if not class_weight: class_weight = None
        if not solver: solver = 'lbfgs'
        if not max_iter: max_iter = 200

        self.model = LogisticRegression(penalty=penalty, C=c, class_weight=class_weight,solver=solver, max_iter=max_iter)

    def train(self, x, y, **kwargs):
        """
        Training the model and saving the history of training
        """
        print("Training model")
        self.history = self.model.fit(x,y)
        crosvalscore = cross_val_score(self.model, x, y, cv=5, scoring='f1')
        print(crosvalscore)

    def test(self, x,y, **kwargs):
        print("Testing model")
        y_pred = self.model.predict(x)
        score = f1_score(y, y_pred)
        return(score)

    def save(self, overwrite=True, **kwargs):
        print("Saving model")
        path = models_store_path+self.name
        pickle.dump(self, open(path, 'wb'))
        joblib.dump(self, 'ourLR.pkl')

    def load(self, **kwargs):
        print("Loading model")
        path = models_store_path+self.name
        self.model = joblib.load('ourLR.pkl')




