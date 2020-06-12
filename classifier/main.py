import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from embedding import zero_matrix_train_location, zero_matrix_test_location, \
    zero_matrix_full_train_location, replaced_zero_matrix_full_train_location, \
    replaced_zero_matrix_test_location
from classifier.pipeline import run_train_pipeline



if __name__ == "__main__":

    build_params = {
        "cell_type":"GRU",
        "num_layers":1,
        "hidden_size":64,
        "optimizer":"adam",
        "dropout_rate":0.4,
        "train_embedding":False,
        "use_normalization":True,
        "use_attention":True,
        "heads":5,
        "penalization":False,
        "gamma":0.3,
        "use_convolution":False,
        "dilation_rate":1
    }
    train_params = {"epochs":10,
                    "batch_size":1024,
                    "validation_split":0.2}

    run_train_pipeline("recurrent_NN",
                       "Attention_GRU_5heads_full",
                       load_model=False,
                       prediction_mode=False,
                       data_location=replaced_zero_matrix_full_train_location,
                       cv_on=False,
                       choose_randomly=True,
                       random_percentage=0.1,
                       test_data_location=None,
                       build_params = build_params,
                       train_params=train_params,
                       generator_mode=False)