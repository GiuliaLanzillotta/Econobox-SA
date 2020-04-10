# All necessary tools to train a glove embedding 
from __init__ import cooc_location, settings_location, \
    vocab_location, glove_embedding_location
import numpy as np 
import pickle
import json
import os

#TODO: build a class Embedder and a class GloVe that extends the Embedder.

def load_glove(text, vocab, embedding_location=glove_embedding_location):
    """
    Loading glove embedding. 
    This function transforms the given text in a sequence of 
    embedding vectors. 
    @param text: list(int)
        The list of indices of the words in the vocabulary.
    @param vocab: dict
        The vocabulary in use. 
    @param embedding_location: str
        The path to the glove embedding file to use.
    """
    abs_path = os.path.abspath(os.path.dirname(__file__))



def train_glove(nrows,n_emb=1,embedding_location=glove_embedding_location,\
    epochs=10,eta=e-3):
    """
    Training GloVe embeddings on nrows tweets.
    @param nrows: int 
        Number of tweets.
    @param nemb: int
        Number of embedding matrices to build.
        It should be either 1 or 2. 
        If 2 is selected than GloVe will be trained to build 
        a focus matrix and a context matrix.
    @param embedding_location: str
        Relative path to the output embedding file.
    @param epochs: int
        Number of training epochs.
    @param eta: float
        Learning rate.
    @returns: np.ndarray
        It will return the embedding matrices.
        Note that if n_emb == 2 then it will return both embeddings.
    """
    abs_path = os.path.abspath(os.path.dirname(__file__))
    # read glove hyperparameters from settings
    print("Loading hyperparameters")
    settings = json.loads(os.path.join(abs_path,settings_location))
    BETA = settings["glove_beta"]
    ALPHA = settings["glove_alpha"]
    EMBEDDING_DIM = settings["embedding_dim"]
    MAX = BETA*nrows
    # read co-occurrence matrix 
    print("Opening co-occurrence matrix")
    with open(cooc_location, 'rb') as f:
        cooc = pickle.load(f)
    # Initialise the embeddings
    xs = np.random.normal(size=(cooc.shape[0], EMBEDDING_DIM))
    if n_emb == 2: ys = np.random.normal(size=(cooc.shape[1], EMBEDDING_DIM))
    else: ys = xs
    # start glove training 
    print("Started GloVe training")
    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / MAX) ** ALPHA)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    # save in np binary format
    np.savez(os.path.join(abs_path,embedding_location), xs, ys)
    if n_emb == 2: return xs,ys
    else: return xs