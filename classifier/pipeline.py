# training and prediction pipeline should be implemented here
from classifier.vanilla_NN import vanilla_NN
from classifier.recurrent_NN import recurrent_NN
from classifier.SVM_classi import SVM_classi
from classifier.LR_classi import LR_classi
from preprocessing import standard_vocab_name
from preprocessing.tokenizer import get_vocab_dimension
from embedding import matrix_train_location, glove_embedding_location
import embedding
import numpy as np
import os
from sklearn.model_selection import KFold
from classifier import K_FOLD_SPLITS

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
                      train_params=None,
                      **kwargs):
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


def get_recurrent_model(model_name,
                        embedding_dim=embedding.embedding_dim,
                        train_data=None,
                        load_model=False,
                        train_model=False,
                        save_model=False,
                        test_data=None,
                        build_params=None,
                        train_params=None,
                        **kwargs):
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
            ```
            >>> Example : build_params = {"activation":'relu', \
                                            "optimizer":'adam',\
                                            "loss":"binary_crossentropy",\
                                            "metrics":None,\
                                            "cell_type":"LSTM",\
                                            "num_layers":3,\
                                            "hidden_size":64,\
                                            "train_embedding":False,\
                                            "use_attention":False}
            ```
    :param train_params: (dict) dictionary of parameters to pass to build the model
            >>> Example : {"epochs":10, \
                            "batch_size":32, \
                            "validation_split":0.2}

    :param kwargs: additional arguments
        Arguments accepted:
        - :arg load_embedding: (bool) whether to load an embedding matrix into the classifier
            (if false, the classifier will learn the embedding from scratch)
        - :arg embedding_location: (str) - only used if the above parameter is true- path to the
            file that stores the embedding matrix
        - :arg vocabulary: (str) vocabulary in use
    :return: an instance of Vanilla_NN class
    """

    vocabulary = kwargs.get("vocabulary")
    if not vocabulary: vocabulary = standard_vocab_name
    vocab_dim = get_vocab_dimension(vocabulary)
    # --------------------
    # Opening pre-trained embedding matrix
    load_embedding = kwargs.get("load_embedding")
    embedding_location = kwargs.get("embedding_location")
    if not embedding_location: embedding_location = glove_embedding_location
    embedding_matrix = None
    if load_embedding:
        abs_path = os.path.abspath(os.path.dirname(__file__))
        embedding_matrix = np.load(os.path.join(abs_path, embedding_location))['arr_0']
    # -------------------
    # Building the model
    recurrent = recurrent_NN(embedding_dimension=embedding_dim,
                             vocabulary_dimension=vocab_dim,
                             name = model_name,
                             embedding_matrix=embedding_matrix)
    recurrent.build(**build_params)
    if load_model: recurrent.load()
    # ----------------
    # Training, testing and saving
    if train_model:
        x_train = train_data[:, 0:-1]
        y_train = train_data[:, -1]
        recurrent.train(x_train, y_train, **train_params)
    if test_data is not None:
        x_test = test_data[:, 0:-1]
        y_test = test_data[:, -1]
        recurrent.test(x_test,y_test, **train_params)
    if save_model: recurrent.save()
    return recurrent

def get_LR_model(model_name,
                 embedding_dim=embedding.embedding_dim,
                 train_data=None,
                 load_model=False,
                 train_model=True,
                 save_model=True,
                 test_data=None,
                 build_params=None,
                 train_params=None,
                 **kwargs):
    """
          Creates a new instance of LR_classi
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
                  Example : {c:1,
                            solver:'lbfgs',
                            }
          :param train_params: (dict) dictionary of parameters to pass to build the model
                  Example : {validation_split:0.2}
          :return: an instance of LR_classi class
          """
    ourLR = LR_classi(embedding_dim, model_name)
    ourLR.build(**build_params)
    if load_model: ourLR.load()
    if train_model:
        x_train = train_data[:, 0:-1]
        y_train = train_data[:, -1]
        ourLR.train(x_train, y_train)
    if test_data is not None:
        x_test = test_data[:, 0:-1]
        y_test = test_data[:, -1]
        ourLR.test(x_test, y_test)
    if save_model: ourLR.save()
    return ourLR


def get_SVM_model(model_name,
                  embedding_dim=embedding.embedding_dim,
                  train_data=None,
                  load_model=False,
                  train_model=True,
                  save_model=True,
                  test_data=None,
                  build_params=None,
                  train_params=None,
                  **kwargs):
    """
       Creates a new instance of SVM_classi
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
               Example : {c:1,
                         kernel:'rbf',
                         }
       :param train_params: (dict) dictionary of parameters to pass to build the model
               Example : {validation_split:0.2}
       :return: an instance of SVM_classi class
       """
    oursvm = SVM_classi(embedding_dim, model_name)
    oursvm.build(**build_params)
    if load_model: oursvm.load()
    if train_model:
        x_train = train_data[:, 0:-1]
        y_train = train_data[:, -1]
        oursvm.train(x_train, y_train)
    if test_data is not None:
        x_test = test_data[:, 0:-1]
        y_test = test_data[:, -1]
        oursvm.test(x_test, y_test)
    if save_model: oursvm.save()
    return oursvm


def cross_validation(train_data, typemodel, embedding_dim, model_name):

    typemodeldict = {
        "vanilla_NN": vanilla_NN(embedding_dim, model_name),
        "SVM_classi": SVM_classi(embedding_dim, model_name),
        "LR_classi": LR_classi(embedding_dim, model_name),
        "recurrent_NN": recurrent_NN(embedding_dim,model_name),
        "_": lambda: None  # empty function
    }

    ourmodel = typemodeldict[typemodel]

    build_params = None
    train_params = None
    x_t = train_data[:, 0:-1]
    y_t = train_data[:, -1]
    kf = KFold(n_splits=K_FOLD_SPLITS, shuffle=True)
    mses = []
    ourmodel.build()

    for train_index, test_index in kf.split(x_t):
        print(test_index)
        X_train, X_test, y_train, y_test = x_t[train_index], x_t[test_index], y_t[train_index], y_t[test_index]
        ourmodel.train(X_train, y_train)
        score = ourmodel.test(X_test,y_test)
        mses.append(score)
        print(score)

    return np.mean(mses), np.std(mses)

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
                       train_params=None,
                       train_embedding=False):
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
        "recurrent_NN":get_recurrent_model,
        "SVM_classi": get_SVM_model,
        "LR_classi": get_LR_model,
        "": lambda : None # empty function
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
                     train_params=train_params,
                     # Nota bene:
                     # don't worry about the next arguments
                     # if you're not using the recurrent net :
                     # they will be automatically ignored by all other functions
                     load_embedding=True,
                     embedding_location = glove_embedding_location)
    #cross_validation(train_data=train_matrix, typemodel='LR_classi', embedding_dim=embedding.embedding_dim, model_name='ourLR')
    return model


