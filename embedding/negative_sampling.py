import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)


# All necessary tools to train a glove embedding
from abc import abstractmethod, ABC

from embedding import settings_location, embedding_dim, \
    stanford_embedding_location
from embedding.embedding_base import EmbeddingBase
from data import sample_dimension
from preprocessing import cooc_folder, vocabularies_folder
from scipy.special import expit
import numpy as np 
import pickle
import json
import os
import math


class NegSamplingEmbedding(EmbeddingBase):
    """Implements GloVe version of Embedding"""

    def __init__(self,
                 embedding_name,
                 embedding_dimension,
                 vocabulary,
                 cooc,
                 neg_cooc,   
                 input_dimension = sample_dimension,
                 load = False):
        super(NegSamplingEmbedding, self).__init__(embedding_name,
                                             embedding_dimension,
                                             vocabulary,
                                             cooc,
                                             load)
        self.neg_cooc = neg_cooc

    def eval_opt_function(self, cooc, neg_cooc, xs):
        L = 0
        for ix, jy, a, b in zip(cooc.row, cooc.col, cooc.data, neg_cooc.data):
            x, y = xs[ix, :].copy(), xs[jy, :].copy()
            L += a * math.log(expit(np.dot(x, y))) + b * math.log(expit(-np.dot(x, y)))

        return L

    def train_embedding(self, epochs, eta):
        """
        Training NegSampling embedding.
        This method uses SGD.
        :param epochs: number of epochs to train for
        :param eta: learning rate
        :return: the trained embedding matrix
        """
        # read co-occurrence matrix
        print("Opening co-occurrence matrix")
        with open(cooc_folder+self.cooc, 'rb') as f:
            cooc = pickle.load(f)

        with open(cooc_folder+self.neg_cooc, 'rb') as f:
            neg_cooc = pickle.load(f)
        # start glove training
        xs = self.embedding_matrix
        print("Started NegSampling training")
        print(epochs)
        counter = 0
        b = True

        L = self.eval_opt_function(cooc, neg_cooc, xs)
        print(L)

        for epoch in range(epochs):
            #xs_n = xs.copy()
            print("epoch {}".format(epoch))
            for ix, jy, a, b in zip(cooc.row, cooc.col, cooc.data, neg_cooc.data):
                #print(ix, jy)
                # the indices in the embedding matrix equal the
                # indices in the vocabulary + 1
                # because we want to leave the 0 position free
                ix +=1
                jy +=1
                
                x, y = xs[ix, :].copy(), xs[jy, :].copy()
                
                cooc_scale = a * expit(-np.dot(x, y))
                neg_cooc_scale = b * expit(np.dot(x, y))
        
                xs[ix, :] += eta * (cooc_scale - neg_cooc_scale) * y 
                xs[jy, :] += eta * (cooc_scale - neg_cooc_scale) * x


                L -= a * math.log(expit(np.dot(x, y))) + b * math.log(expit(-np.dot(x, y)))
                x_n, y_n = xs[ix, :].copy(), xs[jy, :].copy()
                L += a * math.log(expit(np.dot(x_n, y_n))) + b * math.log(expit(-np.dot(x_n, y_n)))
               

            print(L)          
    
              
        self.embedding_matrix = xs

