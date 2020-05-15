# Main of the embedding module
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from embedding.pipeline import run_embedding_pipeline
from embedding import zero_matrix_train_location


if __name__ == "__main__":
    run_embedding_pipeline(no_embedding=True,

                           output_location=zero_matrix_train_location)
    exit(0)