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

        build_params = {"optimizer": 'adam',
                        "metrics": ['accuracy'],
                        "adapter_size": 1,
                        "dropout_rate": 0.5,
                        "max_seq_length": 128}

        train_params = {"epochs": 10,
                        "batch_size": 32,
                        "validation_split": 0.2,
                        "use_categorical": False}

        run_train_pipeline("BERT_NN",
                           "BERT_pred1",
                           load_model=False,
                           prediction_mode=False,
                           text_data_mode_on=True,
                           data_location=tweetDF_location,
                           max_seq_length=128,
                           cv_on=False,
                           choose_randomly=False,
                           random_percentage=0.3,
                           test_data_location=None,
                           build_params=build_params,
                           train_params=train_params,
                           generator_mode=False)
