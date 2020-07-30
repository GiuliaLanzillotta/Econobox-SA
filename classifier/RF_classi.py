from classifier.classifier_base import ClassifierBase
from classifier import models_store_path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import _joblib
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from embedding.pipeline import generate_training_matrix
from embedding.pipeline import get_glove_embedding
import os

class RF_classi(ClassifierBase):
    """Random Forest classifier"""

    def __init__(self,
                 embedding_dimension,
                 name="ourRF"):
        super().__init__(embedding_dimension, name=name)
        self.history = None
        self.model = None

    def build(self, **kwargs):
        print("Building model.")

        n_estimators = kwargs.get("n_estimators")
        criterion = kwargs.get("criterion")
        max_depth = kwargs.get("max_depth")
        min_samples_split = kwargs.get("min_sample_split")
        min_samples_leaf = kwargs.get("min_samples_leaf")
        min_weight_fraction_leaf = kwargs.get("min_weight_fraction_leaf")
        max_features = kwargs.get("max_features")
        bootstrap = kwargs.get("bootstrap")
        warm_start = kwargs.get("warm_start")
        class_weight = kwargs.get("class_weight")

        if not n_estimators: n_estimators = 100
        if not criterion: criterion = "gini"
        if not max_depth: max_depth = None
        if not min_samples_split: min_samples_split = 2
        if not min_samples_leaf: min_samples_leaf = 1
        if not min_weight_fraction_leaf : min_weight_fraction_leaf = 0.0
        if not max_features : max_features = "auto"
        if not bootstrap : bootstrap = True
        if not warm_start : warm_start = False

        self.model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                            bootstrap=bootstrap, warm_start=warm_start)

    def train(self, x, y, **kwargs):
        """
        Training the model and saving the history of training
        """
        generator_mode = kwargs.get("generator_mode",False)
        print("Training model")

        if generator_mode:
            embedding = get_glove_embedding(vocabulary_file="full_vocab_in_stanford.pkl",
                                            load_from_file=True,
                                            load_Stanford=False,
                                            file_name="necessary_stanford.npz",
                                            train=False,
                                            save=True)
            for chunk in generate_training_matrix(embedding=embedding):
                self.history = self.model.fit(chunk[0],chunk[1])


        else:
            self.history = self.model.fit(x,y)
            crossvalscore = cross_val_score(self.model,x,y, cv=5)
            print(crossvalscore)

    def test(self, x,y, **kwargs):
        print("Testing model")
        y_pred = self.model.predict(x)
        score = f1_score(y, y_pred)
        return(score)

    def make_predictions(self, x, save=True, **kwargs):
        print("Making predictions")
        preds = self.model.predict(x)
        preds[preds == 0] = -1
        if save: self.save_predictions(preds)

    def save(self, overwrite=True, **kwargs):
        print("Saving model")
        abs_path = os.path.abspath(os.path.dirname(__file__))
        path = models_store_path + self.name
        _joblib.dump(self.model, os.path.join(abs_path, path))

    def load(self, **kwargs):
        abs_path = os.path.abspath(os.path.dirname(__file__))
        print("Loading model")
        path = models_store_path + self.name
        self.model = _joblib.load(os.path.join(abs_path, path))
