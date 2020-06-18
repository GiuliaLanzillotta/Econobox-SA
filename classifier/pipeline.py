# training and prediction pipeline should be implemented here
from classifier.vanilla_NN import vanilla_NN
from classifier.BERT_NN import BERT_NN
from classifier.recurrent_NN import recurrent_NN, attention_NN
from classifier.convolutional_NN import convolutional_NN
from classifier.SVM_classi import SVM_classi
from classifier.ET_NN import etransformer_NN
from classifier.LR_classi import LR_classi
from classifier.RF_classi import RF_classi
from classifier.Adaboost_classi import Adaboost_classi
from preprocessing import standard_vocab_name
from preprocessing.tokenizer import get_vocab_dimension,load_inverse_vocab
from embedding.pipeline import get_glove_embedding, generate_training_matrix, get_validation_data
from embedding import matrix_train_location, embeddings_folder, matrix_test_location
from embedding import bert_matrix_train_location
from embedding.sentence_embedding import  no_embeddings
from data import replaced_train_negative_location, replaced_train_positive_location, \
    full_dimension
from data import tweetDF_location
from data import testDF_location
import embedding
import numpy as np
import os
np.random.seed(42)
from sklearn.model_selection import KFold
from classifier import K_FOLD_SPLITS
from classifier.BERT_NN import PP_BERT_Data
from preprocessing.tweetDF import load_tweetDF
#from preprocessing.tweetDF import load_testDF

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

def get_BERT_model(model_name,
                   embedding_dim = embedding.embedding_dim,
                   max_seq_length = 128,
                   train_data=None,
                   load_model=False,
                   train_model=False,
                   save_model=False,
                   test_data=None,
                   build_params=None,
                   train_params=None,
                   **kwargs):
    """
        Creates a new instance of BERT_NN
        Parameters:
        :param model_name: (str) the name of the model to create, or the name of the model to load.
        :param embedding_dim: (int) dimension of the embedding space
        (this is actually not needed, but because this model inherits from base_NN we need it)
        :param train_data : pandas dataframe (with the tweets). Existing of two colums
        colum1= the tweets, column2 = the sentiment
        :param load_model: (bool) whether to load the model from file.
        :param train_model: (bool) whether to train the model.
        :param save_model: (bool) whether to save the model
        :param test_data: (np.ndarray) if not None, the model will be tested against this test data.
        :param build_params: (dict) dictionary of parameters to pass to build the model
                Example : {optimizer:'adam',
                          metrics:accuracy,
                          adapter_size: 64,
                          dropout_rate = 0.5}
        Note that if adapter_size = 0, all of the parameters of BERT are fine tuned
        :param train_params: (dict) dictionary of parameters to pass to build the model
                Example : {epochs:10,
                            batch_size:32,
                            validation_split:0.2}
        :return: an instance of BERT_NN class
        """

    ourBERT = BERT_NN(max_seq_length,embedding_dim,model_name)
    ourBERT.build(**build_params)
    if load_model: ourBERT.load()
    if train_model:
        x_train = train_data.train_x
        y_train = train_data.train_y
        ourBERT.train(x_train, y_train, **train_params)
    if test_data is not None:
        x_test = test_data[:, 0:-1]
        y_test = test_data[:, -1]
        ourBERT.test(x_test, y_test, **train_params)
    if save_model: ourBERT.save()
    return ourBERT




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
    Creates a new instance of recurrent_NN
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
                                            "loss":"binary_crossentropy",\
                                            "metrics":None,\
                                            "cell_type":"LSTM",\
                                            "num_layers":3,\
                                            "hidden_size":64,\
                                            "train_embedding":False,\
                                            "use_attention":False, \
                                            "optimizer":"rmsprop"}
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
    embedding_name = kwargs.get("embedding_location","glove_emb.npz")
    generator_mode = kwargs.get("generator_mode", False)
    max_len = kwargs.get("max_len",100)
    if load_embedding:
        glove_embedding = get_glove_embedding(vocabulary_file=vocabulary,
                                              load_from_file=True,
                                              load_Stanford=False, #no need to reload the stanford embedding when we already load the embedding matrix from file
                                              file_name=embedding_name,
                                              train=False,
                                              save=False)
        embedding_matrix = glove_embedding.embedding_matrix
    # -------------------
    # Building the model
    use_attention = build_params.get("use_attention")
    recurrent_fun = recurrent_NN
    if use_attention: recurrent_fun = attention_NN
    recurrent = recurrent_fun(embedding_dimension=embedding_dim,
                              vocabulary_dimension=vocab_dim,
                              name = model_name,
                              embedding_matrix=embedding_matrix)
    recurrent.build(**build_params)
    if load_model:recurrent.load()
    # ----------------
    # Training, testing and saving
    if train_model:
        x_train, y_train = None, None
        if not generator_mode:
            x_train = train_data[:, 0:-1]
            y_train = train_data[:, -1]

        generator_params = {
            "embedding": glove_embedding,
            "input_files":[replaced_train_negative_location,replaced_train_positive_location],
            "input_entries":full_dimension,
            "max_len":max_len
        }

        recurrent.train(x_train, y_train,
                        generator_mode=generator_mode,
                        **generator_params,
                        **train_params)
    if save_model: recurrent.save()
    if test_data is not None:
        idx2word = load_inverse_vocab(vocabulary)
        x_test = test_data[:, 0:-1]
        y_test = test_data[:, -1]
        recurrent.test(x_test,y_test, idx2word=idx2word)
    # ---------------
    # Visualization
    visualize_attention = kwargs.get("visualize_attention", train_model)
    sentence_pos = "I'm loving this project, let's keep on working guys!"
    sentence_neg = "I hate bugs, but not as much as I hate cooking."
    if visualize_attention and use_attention:
        #Note: visualization can only be used with the attention model
        # 1. get the vectorised representation of the sentence
        sentence_pos_vec = no_embeddings(sentence_pos, embedding=glove_embedding)
        sentence_neg_vec = no_embeddings(sentence_neg, embedding=glove_embedding)
        # 2. get the attention plot
        recurrent.visualize_attention(sentence_pos, sentence_pos_vec)
        recurrent.visualize_attention(sentence_neg, sentence_neg_vec)

    return recurrent


def get_convolutional_model(model_name,
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
    Creates a new instance of convolutional_NN
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
    :param train_params: (dict) dictionary of parameters to pass to build the model
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
    embedding_name = kwargs.get("embedding_location","glove_emb.npz")
    generator_mode = kwargs.get("generator_mode", False)
    if load_embedding:
        glove_embedding = get_glove_embedding(vocabulary_file=vocabulary,
                                              load_from_file=True,
                                              load_Stanford=False, #no need to reload the stanford embedding when we already load the embedding matrix from file
                                              file_name=embedding_name,
                                              train=False,
                                              save=False)
        embedding_matrix = glove_embedding.embedding_matrix
    # -------------------
    # Building the model
    convolutional_fun = convolutional_NN
    convolutional = convolutional_fun(embedding_dimension=embedding_dim,
                                      vocabulary_dimension=vocab_dim,
                                      name = model_name,
                                      embedding_matrix=embedding_matrix)
    convolutional.build(**build_params)
    if load_model:convolutional.load()
    # ----------------
    # Training, testing and saving
    if train_model:
        x_train, y_train = None, None
        if not generator_mode:
            x_train = train_data[:, 0:-1]
            y_train = train_data[:, -1]

        convolutional.train(x_train, y_train,
                            generator_mode=generator_mode,
                            **train_params)

    if save_model: convolutional.save()
    if test_data is not None:
        idx2word = load_inverse_vocab(vocabulary)
        x_test = test_data[:, 0:-1]
        y_test = test_data[:, -1]
        convolutional.test(x_test,y_test, idx2word=idx2word)

    return convolutional


def get_ET_model(model_name,
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
    Creates a new instance of convolutional_NN
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
    :param train_params: (dict) dictionary of parameters to pass to build the model
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
    embedding_name = kwargs.get("embedding_location","glove_emb.npz")
    generator_mode = kwargs.get("generator_mode", False)
    #TODO: handle more than one embedding
    if load_embedding:
        glove_embedding = get_glove_embedding(vocabulary_file=vocabulary,
                                              load_from_file=True,
                                              load_Stanford=False, #no need to reload the stanford embedding when we already load the embedding matrix from file
                                              file_name=embedding_name,
                                              train=False,
                                              save=False)
        embedding_matrix = glove_embedding.embedding_matrix
    # -------------------
    # Building the model
    etransformer = etransformer_NN(embedding_dimension=embedding_dim,
                                   vocabulary_dimension=vocab_dim,
                                   embedding_matrices=embedding_matrix,
                                   number_of_embeddings=1,
                                   name=model_name)
    etransformer.build(**build_params)
    if load_model:etransformer.load()
    # ----------------
    # Training, testing and saving
    if train_model:
        x_train, y_train = None, None
        if not generator_mode:
            x_train = train_data[:, 0:-1]
            y_train = train_data[:, -1]

        etransformer.train(x_train, y_train,
                            generator_mode=generator_mode,
                            **train_params)

    if save_model: etransformer.save()
    if test_data is not None:
        idx2word = load_inverse_vocab(vocabulary)
        x_test = test_data[:, 0:-1]
        y_test = test_data[:, -1]
        etransformer.test(x_test,y_test, idx2word=idx2word)

    return etransformer



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

def get_RandomForest_model(model_name,
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
       Creates a new instance of RF_classi
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
       :return: an instance of RF_classi class
       """
    ourRF = RF_classi(embedding_dim, model_name)
    ourRF.build(**build_params)
    if load_model: ourRF.load()
    if train_model:
        x_train = train_data[:, 0:-1]
        y_train = train_data[:, -1]
        ourRF.train(x_train, y_train)
    if test_data is not None:
        x_test = test_data[:, 0:-1]
        y_test = test_data[:, -1]
        ourRF.test(x_test, y_test)
    if save_model: ourRF.save()
    return ourRF

def get_Adaboost_model(model_name,
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
       Creates a new instance of RF_classi
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
       :return: an instance of RF_classi class
       """
    ourADA = Adaboost_classi(embedding_dim, model_name)
    ourADA.build(**build_params)
    if load_model: ourADA.load()
    if train_model:
        x_train = train_data[:, 0:-1]
        y_train = train_data[:, -1]
        ourADA.train(x_train, y_train)
    if test_data is not None:
        x_test = test_data[:, 0:-1]
        y_test = test_data[:, -1]
        ourADA.test(x_test, y_test)
    if save_model: ourADA.save()
    return ourADA

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


def cross_validation(train_data, model_fun,
                     embedding_dim, model_name,
                     build_params, train_params):

    ourmodel = model_fun(embedding_dim,model_name)
    x_t = train_data[:, 0:-1]
    y_t = train_data[:, -1]
    kf = KFold(n_splits=K_FOLD_SPLITS, shuffle=True)
    mses = []
    ourmodel.build(build_params)

    for train_index, test_index in kf.split(x_t):
        print(test_index)
        X_train, X_test, y_train, y_test = x_t[train_index], x_t[test_index], y_t[train_index], y_t[test_index]
        ourmodel.train(X_train, y_train, train_params)
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
                       prediction_mode=False,
                       load_model=False,
                       text_data_mode_on = True,
                       max_seq_length = 128,
                       choose_randomly=False,
                       random_percentage=0.1,
                       data_location=testDF_location,
                       generator_mode=False,
                       cv_on=False,
                       test_data_location=None,
                       build_params=None,
                       train_params=None):
    """
    By default, this function created a new instance of 'model_type' model,  trains it
    from scratch (no loading) and saves it.
    :param random_percentage:
    :param choose_randomly: whether to work on a random subset of the dataset.
        ( Use when working with the full data matrix).
        :type choose_randomly: bool
    :param max_seq_length: (Only used when text_data_mode_on is True)
        The maximum length of the sentence.
        :type max_seq_length : int
    :param text_data_mode_on: whether to load the data as pure text.
        :type text_data_mode_on : bool
    :param generator_mode: whether to use a generator for the input instead of a matrix.
        :type generator_mode: bool
    :param model_type: (str). Should appear as a key in 'model_pipeline_fun'
    :param model_name: (str). Name of the model, very important if you want to load it from file.
    :param prediction_mode: (bool). Whether to load a classifier in prediction mode
            If True, a classifier will be loaded (no training) and
            a new submission file will be created.
    :param load_model: (bool). Whether to load a pre-trained model (False by default).
    :param data_location: (str/path). Path to the training matrix or the prediction input matrix.
    :param cv_on: (bool) Whether to use or not cross validation to assess the model.
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
        "Adaboost_classi":get_Adaboost_model,
        "RF_classi": get_RandomForest_model,
        "BERT_NN": get_BERT_model,
        "convolutional_NN":get_convolutional_model,
        "ET_NN":get_ET_model,
        "": lambda : None # empty function
    }

    abs_path = os.path.abspath(os.path.dirname(__file__))

    ## DATA LOADING
    data_matrix = None
    test_matrix = None
    if text_data_mode_on:
        data_matrix = PP_BERT_Data(load_tweetDF(), classes=[0,1], max_seq_length=max_seq_length,
                                   prediction_mode=prediction_mode)
        if prediction_mode:
            data_matrix = PP_BERT_Data(load_testDF(),classes=[0,1], max_seq_length=max_seq_length,
                                   prediction_mode=prediction_mode )
            data_matrix = data_matrix.pred_x
    if not text_data_mode_on and not generator_mode:
        print("Loading data")
        data_matrix = np.load(os.path.join(abs_path, data_location))['arr_0']
        if choose_randomly and not prediction_mode:
            train_size = int(random_percentage * data_matrix.shape[0])
            test_size = int(0.1 * train_size)
            print("Randomly picking ", train_size, " indices to train on and ", test_size, " indices to test on.")
            random_indices = np.random.choice(data_matrix.shape[0], train_size + test_size, replace=False)
            train_indices = random_indices[:train_size]
            test_indices = random_indices[train_size:]
            test_matrix = data_matrix[test_indices, :]
            data_matrix = data_matrix[train_indices, :]
    if test_data_location is not None:
        test_matrix = np.load(os.path.join(abs_path, test_data_location))['arr_0']

    ## MODEL FUNCTIONS
    function = model_pipeline_fun[model_type]
    if cv_on: cross_validation(train_data=data_matrix, model_fun=function,
                               embedding_dim=embedding.embedding_dim, model_name=model_name,
                               build_params=build_params, train_params=train_params)
    model = function(model_name,
                    train_data=data_matrix,
                    load_model=load_model,
                    train_model=not (prediction_mode or cv_on),
                    save_model=not (prediction_mode or cv_on),
                    test_data=test_matrix,
                    build_params=build_params,
                    train_params=train_params,
                    # Nota bene:
                    # don't worry about the next arguments
                    # if you're not using the recurrent net :
                    # they will be automatically ignored by all other functions
                    vocabulary="full_vocab_in_stanford.pkl",
                    load_embedding=True,
                    embedding_location ="necessary_stanford.npz",
                    generator_mode=generator_mode,
                    max_len=100,
                    # parameter for BERT
                    max_seq_length= max_seq_length)
    if prediction_mode:
       model.make_predictions(data_matrix, save=True)
    return model


