# Tools to build the co-occurrence matrix from the input files and the vocabulary
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from data import train_positive_location, train_negative_location, \
  train_positive_sample_location, train_negative_sample_location
from preprocessing import cooc_folder
from preprocessing.tokenizer import tokenize_text, load_vocab
from scipy.sparse import *
import random
import pickle
import numpy as np
import math

table_size = 10**3 


#computes probabilities for sampling according to unigram distributeion to the power 3/4
def compute_sampling_probabilities(vocab, input_files):
  occurences = [0]*len(vocab)   
  counter = 1
  # opening each file
  abs_path = os.path.abspath(os.path.dirname(__file__))
  for fn in input_files:
    with open(os.path.join(abs_path, fn), encoding="utf8") as f:
      print("Working on ",fn)
      # looking at each line
      for line in f:
        # Here we filter out the words that are not in the vocabulary 
        words = tokenize_text(line)
        tokens = [vocab.get(t, -1) for t in words]
        tokens = [t for t in tokens if t >= 0]
        ll = len(tokens) # filtered line length
          
        for j in range(ll):   
          t = tokens[j]
          occurences[t] += 1
             

          if counter % 10000 == 0:
              print(counter)
          counter += 1
  
  probabilities = [n**(0.75) for n in occurences]
  Z = sum(probabilities)  
  probabilities = [p/Z for p in probabilities]

  return probabilities        


# build a negative cooc for negative sampling word2vec
def build_neg_cooc(vocab_name, num_samples,
               window_size=None,
               weighting="None",
               output_name="neg_cooc.pkl",
               input_files=None):
  """ 
  Parameters: 
  - vocab_name: the name of the .pkl file containing the vocabulary to use
  - window_size : if None, using the whole line as a window, otw a number is expected. 
    Note: the window size cannot be larger tha the line length.
  - weighting: only 2 types supported for now, one of 'None' or 'Distance' 
  - output_name: name of the the .pkl file where the negative co-occurrence matrix will be stored
  - input_files: (list(str)) name of the files from which to build the co-occurrence matrix
  -  

  By default the output (negative co-occurence matrix) will be saved in a .pkl file in the cooc directory
  """

  if input_files is None:
    input_files = [train_positive_sample_location,train_negative_sample_location]
  vocab = load_vocab(vocab_name)
  vocab_values = list(vocab.values())
  #print(values[1])

  data, row, col = [], [], []
  counter = 1
  # opening each file
  abs_path = os.path.abspath(os.path.dirname(__file__))
  probabilities = compute_sampling_probabilities(vocab, input_files)  
    
  lookup_table = [0] * table_size

  count = 0
  for t in vocab_values:
    r = math.ceil(table_size * probabilities[t]) 
    start = count
    count += r + 1
    end = count
    if(len(lookup_table) < count):
      k = count - len(lookup_table)
      lookup_table[start:] = [t] * (r-k)
      lookup_table.extend([t]*k)
    else:  
      lookup_table[start:end] = [t] * r 

  for fn in input_files:
      with open(os.path.join(abs_path, fn), encoding="utf8") as f:
        print("Working on ",fn)
        # looking at each line
        for line in f:
          # Here we filter out the wordcs that are not in the vocabulary 
          words = tokenize_text(line)
          tokens = [vocab.get(t, -1) for t in words]
          tokens = [t for t in tokens if t >= 0]
          ll = len(tokens) # filtered line length
          if window_size is None or window_size>=ll:
            delta = ll 
          else: 
            delta = window_size 
          
          for j in range(ll):
              t1 = tokens[j]
              for i in range(-1*delta,delta):
                if j+i<0 or j+i>=ll or i==0:
                  # Note: I exclude the self-co-occurrence 
                  # to save space in memory 
                  continue
                #print("start sampling")
                neg_samples = np.random.choice(vocab_values, num_samples, p = probabilities)
                #print("done sampling")
                c = 1
                if weighting == 'Distance':
                  c = c/i
                data.extend([c]*num_samples)
                row.extend([t1]*num_samples)
                col.extend(neg_samples)

          if counter % 100 == 0:
              print(counter)
          counter += 1
            
  # According to scipy documentation the duplicate indices 
  # entries are not summed automatically
  neg_cooc = coo_matrix((data, (row, col)))
  print("summing duplicates (this can take a while)")
  neg_cooc.sum_duplicates()

  # Saving the output
  with open(cooc_folder+output_name, 'wb') as f:
      pickle.dump(neg_cooc, f, pickle.HIGHEST_PROTOCOL)

def load_neg_cooc(file_name):
  """
      Loads the negative co-occurrence matrix with the given name.
      """
  with open(cooc_folder + file_name, 'rb') as f:
    neg_cooc = pickle.load(f)
  return neg_cooc
