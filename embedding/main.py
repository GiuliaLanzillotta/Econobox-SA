# Main of the embedding module
from embedding import glove_embedding_location, settings_location, embedding_dim
from embedding.tweet_embedding import build_embedding_matrix
from embedding.glove import train_glove
from preprocessing import sample_dimension
import numpy as np
import json
import os

#TODO: add embedding test

def run_train_matrix_construction():
    D = embedding_dim
    N = sample_dimension
    abs_path = os.path.abspath(os.path.dirname(__file__))
    emb = np.load(os.path.join(abs_path,glove_embedding_location))['arr_0']
    build_embedding_matrix(True,N,D,emb)


if __name__ == "__main__":
    print("Building training matrix.")
    run_train_matrix_construction()
    exit(0)