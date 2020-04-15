# Main of the embedding module
from embedding import glove_embedding_location, settings_location, embedding_dim
from embedding.pipeline import run_embedding_pipeline
from embedding.pipeline import get_glove_embedding
from preprocessing import sample_dimension
import numpy as np
import json
import os

#TODO: add embedding test

if __name__ == "__main__":
    run_embedding_pipeline()
    exit(0)