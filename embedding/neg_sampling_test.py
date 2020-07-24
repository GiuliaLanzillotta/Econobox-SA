import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import warnings

from preprocessing.tokenizer import load_vocab
from embedding import embedding_dim
from embedding.negative_sampling import NegSamplingEmbedding


vocabulary_file = "vocab.pkl"
cooc_file = "cooc.pkl"
neg_cooc_file = "neg_cooc.pkl"
train_epochs=20
train_eta=1e-5

warnings.filterwarnings('error')


vocab = load_vocab(vocabulary_file)
file_name = "neg_sampling_emb.npz"
neg_sampling_embedding = NegSamplingEmbedding(file_name,
                                     embedding_dim,
                                     vocab,
                                     cooc_file,
                                     neg_cooc_file)
neg_sampling_embedding.train_embedding(epochs=train_epochs,
                                              eta=train_eta)
neg_sampling_embedding.save_embedding()

