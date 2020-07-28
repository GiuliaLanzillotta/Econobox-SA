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
    input_files = [data.replaced_train_full_negative_location_30,
                   data.replaced_train_full_positive_location_30]

    
    run_embedding_pipeline(embedding_fun="sum_embeddings",
                           prediction_mode=False,
                           input_entries=data.subset_thirty_percent_dimension,
                           input_files=input_files,
                           input_labels=[0,1],
                           output_location=data.train_matrices_folder+"tfidf_sample_matrix")
    
    """
    run_embedding_pipeline(embedding_fun="sum_embeddings",
                           prediction_mode=False,
                           input_entries=data.sample_dimension,
                           input_files=input_files,
                           input_labels=None,
                           output_location=embedding.neg_sampling_matrix_train_location,
                           glove=False,
                           use_tf_idf=False, 
                           embedding="negative_sampling",
                           max_len=200)
    """
    # 768 for roberta, 50 for the no embedding function, 200 Glove
    exit(0)
