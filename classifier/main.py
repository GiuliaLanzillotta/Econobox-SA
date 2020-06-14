import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from embedding import zero_matrix_train_location, zero_matrix_test_location, \
    zero_matrix_full_train_location, replaced_zero_matrix_full_train_location, \
    replaced_zero_matrix_test_location, replaced_zero_matrix_train_location
from data import tweetDF_location
from classifier.pipeline import run_train_pipeline



if __name__ == "__main__":

    build_params = {
                    "cell_type":"GRU",
                    "num_layers":1,
                    "hidden_size":128,
                    "optimizer":"adam",
                    "dropout_rate":0.4,
                    "use_normalization":True,
                    "use_attention":True,
                    "heads":5,
                    "penalization":False,
                    "use_convolution":False,
                    "dilation_rate":1
                }

    train_params = {"epochs":10,
                    "batch_size":2048,
                    "validation_split":0.2,
                    "to_categorical":False}

    run_train_pipeline("recurrent_NN",
                       "GRU_Attention_5heads_full3",
                       load_model=False,
                       prediction_mode=False,
                       text_data_mode_on=False,
                       data_location=replaced_zero_matrix_full_train_location,
                       max_seq_length=128,
                       cv_on=False,
                       choose_randomly=True,
                       random_percentage=0.3,
                       test_data_location=None,
                       build_params = build_params,
                       train_params=train_params,
                       generator_mode=False)

