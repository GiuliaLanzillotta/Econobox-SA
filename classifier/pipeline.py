# training and prediction pipeline should be implemented here
from classifier.vanilla_NN import vanilla_NN
from embedding import matrix_train_location, embedding_dim
import numpy as np
import os

def get_vanilla_model(train_data=None,
                      load_model=False,
                      train_model=False,
                      test_data=None,
                      validation_split=0.2):
    """
    Creates a new instance of Vanilla_NN
    :param train_data:
    :param load_model:
    :param train_model:
    :param test_data:
    :param validation_split:
    :return:
    """



def train_model():
    """ Trains a model on train matrix"""
    print("Loading data")
    abs_path = os.path.abspath(os.path.dirname(__file__))
    train_matrix = np.load(os.path.join(abs_path,matrix_train_location))['arr_0']
    N, D = train_matrix.shape
    x_train = train_matrix[:, 0:-1]
    y_train = train_matrix[:, -1]
    print("Building model")
    model = vanilla_NN(embedding_dim)
    model.compile_model()
    print("Training model")
    model.fit_model(x_train, y_train)
    model.plot_history()
    

