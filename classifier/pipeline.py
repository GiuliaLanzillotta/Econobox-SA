# training and prediction pipeline should be implemented here
from classifier.vanilla_NN import vanilla_NN
from classifier.classifier_base import ClassifierBase
from embedding import matrix_train_location
import embedding
import numpy as np
import os

"""
When you implement a new model, you should also implement a new 'get_[name]_model' function with 
exactly this signature and add the function to the 'model_pipeline_fun' dictionary in the 
'run_train_pipeline' function.
"""
def get_vanilla_model(model_name,
                      embedding_dim=embedding.embedding_dim,
                      train_data=None,
                      load_model=False,
                      train_model=False,
                      save_model=False,
                      test_data=None,
                      build_params=None,
                      train_params=None):
    """
    Creates a new instance of Vanilla_NN
    Parameters:
    :param model_name: (str) the name of the model to create, or the name of the model to load.
    :param embedding_dim: (int) dimension of the embedding space
    :param train_data : (np.ndarray) the training data in a single matrix (like the one produced by
            the embedding.pipeline.build_embedding_matrix method
    :param load_model: (bool) whether to load the model from file.
    :param train_model: (bool) whether to train the model.
    :param save_model: (bool) whether to save the model
    :param test_data: (np.ndarray) if not None, the model will be tested against this test data.
    :param build_params: (dict) dictionary of parameters to pass to build the model
            Example : {activation:'relu',
                      optimizer:'adam',
                      loss:"binary_crossentropy",
                      metrics:None}
    :param train_params: (dict) dictionary of parameters to pass to build the model
            Example : {epochs:10,
                        batch_size:32,
                        validation_split:0.2}
    :return: an instance of Vanilla_NN class
    """

    vanilla = vanilla_NN(embedding_dim,model_name)
    vanilla.build(**build_params)
    if load_model: vanilla.load()
    if train_model:
        x_train = train_data[:, 0:-1]
        y_train = train_data[:, -1]
        vanilla.train(x_train, y_train, **train_params)
    if test_data is not None:
        x_test = test_data[:, 0:-1]
        y_test = test_data[:, -1]
        vanilla.test(x_test,y_test, **train_params)
    if save_model: vanilla.save()
    return vanilla


def cross_validation():
    # TODO: implement cross_validation
    pass

def grid_search():
    # TODO: implement grid_search
    # can only be used if the "get_[name]_model" function supports
    # a dictionary of parameters as input
    pass

def run_train_pipeline(model_type,
                       model_name,
                       load_model=False,
                       train_data_location=matrix_train_location,
                       test_data_location=None,
                       build_params=None,
                       train_params=None):
    """
    By default, this function created a new instance of 'model_type' model,  trains it
    from scratch (no loading) and saves it.
    :param model_type: (str). Should appear as a key in 'model_pipeline_fun'
    :param model_name: (str). Name of the model, very important if you want to load it from file.
    :param load_model: (bool). Whether to load a pre-trained model (False by default).
    :param train_data_location: (str/path).
    :param test_data_location: (str/path).
    :param build_params: (dict). Dictionary of build parameter. The entries depend on your specific model.
    :param train_params: (dict). DIctionary of trainin parameters. // //
    :return: an instance of the model
    """
    if build_params is None:
        build_params = {}
    if train_params is None:
        train_params = {}

    model_pipeline_fun = {
        "vanilla_NN": get_vanilla_model,
        "_": lambda : None # empty function
    }

    print("Loading data")
    abs_path = os.path.abspath(os.path.dirname(__file__))
    train_matrix = np.load(os.path.join(abs_path,train_data_location))['arr_0']
    if test_data_location is not None:
        test_matrix = np.load(os.path.join(abs_path, test_data_location))['arr_0']
    else: test_matrix = None

    function = model_pipeline_fun[model_type]
    model = function(model_name,
                     train_data=train_matrix,
                     load_model=load_model,
                     train_model=True,
                     save_model=True,
                     test_data=test_matrix,
                     build_params=build_params,
                     train_params=train_params)
    return model


