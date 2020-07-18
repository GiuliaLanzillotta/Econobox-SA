from classifier.classifier_base import ClassifierBase
from classifier import models_store_path
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import _joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from preprocessing import TFIDF_preprocessing
from data import train_negative_sample_location, train_positive_sample_location

class NaiveBayes_classi(ClassifierBase):
    """Naive Bayes classifier"""

    def __init__(self,
                 embedding_dimension,
                 name="ourNB"):
        super().__init__(embedding_dimension, name=name)
        self.history = None
        self.model = None

    def build(self, **kwargs):
        print("Building model")

        self.model = MultinomialNB(alpha=1.8)

    def train(self, x,y, **kwargs):
        print(x)
        print(y)
        self.history = self.model.fit(x,y)
        crossvalscore = cross_val_score(self.model, x, y, cv=5, scoring='f1')
        print(crossvalscore)

    def test(self, x,y, **kwargs):
        print("testing model")
        y_pred = self.model.predict(x)
        score =f1_score(y, y_pred)
        return(score)

    def make_predictions(self, x, save=True, **kwargs):
        print("Making predictions")
        preds = self.model.predict(x)
        preds[preds==0] = -1
        if save: self.save_predictions(preds)

    def save(self, overwrite=True, **kwargs):
        print("Saving model")
        path = models_store_path+self.name
        pickle.dump(self, open(path, 'wb'))
        _joblib.dump(self.model, 'ourNB.pkl')



    def load(self, **kwargs):
        print("Loading model")
        path = models_store_path+self.name
        self.model = _joblib.load('ourNB.pkl')

"""
ourNB = NaiveBayes_classi(embedding_dimension=-1)
ourNB.build()
data = TFIDF_preprocessing.get_tfidf_train_data(input_files=[train_positive_sample_location, train_negative_sample_location], random_percentage=1)
print(data[0])
print(data[1])
ourNB.train(x=data[0], y=data[1])
"""