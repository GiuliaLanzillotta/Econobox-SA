import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from classifier.pipeline import run_train_pipeline



if __name__ == "__main__":
    build_params = {
        "cell_type":"GRU",
        "num_layers":1,
        "hidden_size":64
    }
    run_train_pipeline("recurrent_NN",
                       "Recurrent_1L_GRU",
                       load_model=False,
                       test_data_location=None,
                       build_params = build_params,
                       train_params={"epochs":1,
                                     "batch_size":100,
                                     "validation_split":0.3})