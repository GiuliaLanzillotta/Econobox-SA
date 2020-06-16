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
        build_params = {"train_embedding": False,
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
                    }
    
        train_params = {"epochs":10,
                        "batch_size":128,
                        "validation_split":0.2,
                        "use_categorical":True}
    
        run_train_pipeline("convolutional_NN",
                           "convolution_3_pool",
                           load_model=True,
                           prediction_mode=False,
                           text_data_mode_on=False,
                           data_location=replaced_zero_matrix_full_train_location,
                           max_seq_length=1024,
                           cv_on=False,
                           choose_randomly=True,
                           random_percentage=0.3,
                           test_data_location=None,
                           build_params = build_params,
                           train_params=train_params,
                           generator_mode=False)
