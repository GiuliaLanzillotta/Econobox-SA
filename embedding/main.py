# Main of the embedding module
import sys
import os
#from BERT_EMBEDDING import get_BERT_EMBEDDING
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from embedding.pipeline import run_embedding_pipeline
import embedding
import data

if __name__ == "__main__":
    input_files = [data.train_negative_location,
                   data.train_positive_location]
    run_embedding_pipeline(embedding_fun="sum_embeddings",
                           prediction_mode=True,
                           input_entries=data.full_dimension,
                           input_files=input_files,
                           output_location=embedding.matrix_test_location2,
                           embedding="roberta-base",
                           max_len=200) # 768 for roberta, 50 for the no embedding function, 200 Glove
    exit(0)