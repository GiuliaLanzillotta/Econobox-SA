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
        build_params = {"train_embedding":False,
                        "use_pretrained_embedding":True,
                        "cell_type": "GRU",
                        "num_layers": 1,
                        "hidden_size": 64,
                        "optimizer": "adam",
                        "use_convolution":True,
                        "num_conv_layers":6,
                        "threshold_channels":600,
                        "penalization":False,
                        "use_attention":False,
                        "dropout_rate": 0.4,
                        "use_normalization": True}
    
        train_params = {"epochs":10,
                        "batch_size":128,
                        "validation_split":0.2,
                        "use_categorical":True}
    
        run_train_pipeline("recurrent_NN",
                           "GRU_128_conv",
                           load_model=True,
                           prediction_mode=True,
                           text_data_mode_on=False,
                           data_location=replaced_zero_matrix_test_location,
                           max_seq_length=1024,
                           cv_on=False,
                           choose_randomly=False,
                           random_percentage=0.3,
                           test_data_location=None,
                           build_params = build_params,
                           train_params=train_params,
                           generator_mode=False)
