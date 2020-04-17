import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from classifier.pipeline import run_train_pipeline



if __name__ == "__main__":
    run_train_pipeline("vanilla_NN",
                       "VanillaNN1",
                       load_model=True,
                       test_data_location=None,
                       build_params = None,
                       train_params={"epochs":1,
                                     "batch_size":100,
                                     "validation_split":0.3})