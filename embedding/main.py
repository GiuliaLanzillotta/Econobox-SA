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
    input_files = [data.train_negative_sample_location,data.train_negative_sample_location]

    run_embedding_pipeline(embedding_fun="sum_embeddings",
                           prediction_mode=False,
                           input_entries=data.sample_dimension,
                           input_files=input_files,
                           input_labels=[0,1],
                           output_location=data.train_matrices_folder+"tfidf_sample_matrix")
    # 768 for roberta, 50 for the no embedding function, 200 Glove
    exit(0)