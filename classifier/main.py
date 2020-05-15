import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from embedding import zero_matrix_train_location, zero_matrix_test_location
from classifier.pipeline import run_train_pipeline



if __name__ == "__main__":
    build_params = {
        "cell_type":"GRU",
        "num_layers":1,
        "hidden_size":64,
        "optimizer":"adam"
    }
    run_train_pipeline("recurrent_NN",
                       "Recurrent_1L_GRU",
                       load_model=False,
                       prediction_mode=True,
                       data_location=zero_matrix_test_location,
                       test_data_location=None,
                       build_params = build_params,
                       train_params={"epochs":3,
                                     "batch_size":32,
                                     "validation_split":0.3})