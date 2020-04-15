# Here we should implement all the necessary tools to build a sentence embedding
# TODO: incorporate possibility of using 2 different embeddings
from . import matrix_train_location, vocab_location
from preprocessing import train_positive_location, train_negative_location, \
  train_positive_sample_location, train_negative_sample_location
from preprocessing.tokenizer import  tokenize_text
import numpy as np
import pickle




