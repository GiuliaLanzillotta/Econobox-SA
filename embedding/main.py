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
    input_files = [data.replaced_train_full_negative_location[:-4]+"_split{0}".format(s)+".txt" for s in range(6)] +\
                  [data.replaced_train_full_positive_location[:-4]+"_split{0}".format(s)+".txt" for s in range(5)]

    run_embedding_pipeline(embedding_fun="transformer_emb",
                           prediction_mode=False,
                           input_entries=data.full_dimension,
                           input_files=input_files,
                           input_labels=[0 for i in range(6)] + [1 for i in range(5)],
                           output_location=embedding.roberta_full_matrix_train_location,
                           embedding="roberta-base",
                           max_len=200) # 768 for roberta, 50 for the no embedding function, 200 Glove
    exit(0)