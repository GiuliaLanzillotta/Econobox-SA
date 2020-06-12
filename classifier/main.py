import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from embedding import zero_matrix_train_location, zero_matrix_test_location, \
    zero_matrix_full_train_location, replaced_zero_matrix_full_train_location
from data import tweetDF_location
from classifier.pipeline import run_train_pipeline



if __name__ == "__main__":

    build_params = None
    train_params = {"epochs":3,
                    "batch_size":64,
                    "validation_split":0.2}

    run_train_pipeline("BERT_NN",
                       "BERT_NN_1st",
                       load_model=False,
                       prediction_mode=False,
                       text_data_mode_on=True,
                       data_location=tweetDF_location,
                       max_seq_length=128,
                       cv_on=False,
                       test_data_location=None,
                       build_params = build_params,
                       train_params=train_params,
                       generator_mode=False)

