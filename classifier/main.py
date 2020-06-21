import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from embedding import zero_matrix_train_location, zero_matrix_test_location, \
    zero_matrix_full_train_location, replaced_zero_matrix_full_train_location, \
    replaced_zero_matrix_test_location, replaced_zero_matrix_train_location
from data import tweetDF_location
from classifier.pipeline import run_train_pipeline, run_ensemble_pipeline


## ----------------
#Saving here the model specific parameters to pass to the
#model pipeline function.

recurrent_specific_params = {
    "vocabulary":"full_vocab_in_stanford.pkl",
    "load_embedding":True,
    "embedding_location":"necessary_stanford.npz",
    "generator_mode":False,
    "max_len":100
}


bert_specific_params = {
    "max_seq_length":50
}

et_specific_params = {
    "number_of_embeddings":2,
    "vocabularies":["full_vocab_in_stanford.pkl","vocab.pkl"],
    "embedding_locations":["necessary_stanford.npz","glove+stanford.npz"]
}


def ensemble_main():

    models = ["convolutional_NN","recurrent_NN"]
    models_names = ["convolution_3_pool","Attention_GRU_5heads_full"]
    data_locations = [replaced_zero_matrix_full_train_location,
                      replaced_zero_matrix_full_train_location]
    models_build_params = [
        ## CONVOLUTIONAL PARAMETERS
        {
            "train_embedding": False,
            "use_pretrained_embedding": True,
            "use_pooling": True,
            "pooling_type": "max",
            "num_convolutions": 3,
            "window_size": 5,
            "dilation_rate": 1,  # no dilation
            "pool_size": 2,
            "hidden_size": 128,
            "dropout_rate": 0.4,
            "use_normalization": True,
            "optimizer": "adam"
        },
        ## ATTENTION PARAMETERS
        {
            "cell_type":"GRU",
            "num_layers":1,
            "hidden_size":128,
            "optimizer":"adam",
            "dropout_rate":0.4,
            "use_normalization":True,
            "use_attention":True,
            "heads":5, # number extracted from section 4.4.2 of the paper
            "penalization":False
        }
    ]
    models_fun_params = [recurrent_specific_params,
                         recurrent_specific_params]
    run_ensemble_pipeline(models,
                          models_names,
                          data_locations,
                          models_build_params,
                          models_fun_params,
                          random_percentage=0.03,
                          prediction_mode=False,
                          ensemble_name="CONVATT_ensemble")


if __name__ == "__main__":
    ensemble_main()
    """
    build_params = {"optimizer": 'adam',
                    "metrics": ['accuracy'],
                    "adapter_size": 1,
                    "train_embedding": False,
                    "use_pretrained_embedding": True,
                    "num_et_blocks":1,
                    "max_len":50} # maximum length in the sequece

    train_params = {"epochs": 10,
                    "batch_size": 1024,
                    "validation_split": 0.2,
                    "use_categorical": True}

    run_train_pipeline("ET_NN",
                       "ET_2",
                       load_model=False,
                       prediction_mode=False,
                       text_data_mode_on=False,
                       data_location=replaced_zero_matrix_full_train_location,
                       cv_on=False,
                       choose_randomly=True,
                       random_percentage=0.3,
                       test_data_location=None,
                       generator_mode=False,
                       build_params=build_params,
                       train_params=train_params,
                       model_specific_params=et_specific_params)
                       """

