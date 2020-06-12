# Main of the embedding module
import sys
import os
#from BERT_EMBEDDING import get_BERT_EMBEDDING
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from data import full_dimension,train_positive_location,train_negative_location, \
    replaced_train_negative_location, replaced_train_positive_location, replaced_test_location
from embedding.pipeline import run_embedding_pipeline
from embedding import zero_matrix_train_location, zero_matrix_test_location, \
    zero_matrix_full_train_location, replaced_zero_matrix_full_train_location, \
    replaced_zero_matrix_test_location

if __name__ == "__main__":
    #get_BERT_EMBEDDING()
    input_files = [replaced_train_negative_location,replaced_train_positive_location]
    run_embedding_pipeline(no_embedding=True,
                           prediction_mode=False,
                           input_entries=full_dimension,
                           input_files=input_files,
                           output_location=replaced_zero_matrix_full_train_location)
    exit(0)