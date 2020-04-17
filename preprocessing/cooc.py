# Tools to build the co-occurrence matrix from the input files and the vocabulary
from __init__ import train_positive_location, train_negative_location, \
  train_positive_sample_location, train_negative_sample_location, cooc_folder
from preprocessing.tokenizer import tokenize_text, load_vocab
from scipy.sparse import *
import numpy as np
import pickle
import os

def build_cooc(vocab_name,
               window_size=None,
               weighting="None",
               output_name="cooc.pkl",
               input_files=None):
  """ 
  Parameters: 
  - vocab_name: the name of the .pkl file containing the vocabulary to use
  - window_size : if None, using the whole line as a window, otw a number is expected. 
    Note: the window size cannot be larger tha the line length.
  - weighting: only 2 types supported for now, one of 'None' or 'Distance' 
  - output_name: name of the the .pkl file where the co-occurrence matrix will be stored
  - input_files: (list(str)) name of the files from which to build the co-occurrence matrix

  By default the output (co-occurence matrix) will be saved in a .pkl file in the cooc directory
  """

  if input_files is None:
    input_files = [train_positive_sample_location,train_negative_sample_location]

  vocab = load_vocab(vocab_name)

  data, row, col = [], [], []
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
                t2 = tokens[j+i]
                c = 1
                if weighting == 'Distance':
                  c = c/i
                data.append(c)
                row.append(t1)
                col.append(t2)

          if counter % 10000 == 0:
              print(counter)
          counter += 1
            
  # According to scipy documentation the duplicate indices 
  # entries are not summed automatically
  cooc = coo_matrix((data, (row, col)))
  print("summing duplicates (this can take a while)")
  cooc.sum_duplicates()

  # Saving the output
  with open(cooc_folder+output_name, 'wb') as f:
      pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)


def load_cooc(file_name):
  """
      Loads the co-occurrence matrix with the given name.
      """
  with open(cooc_folder + file_name, 'rb') as f:
    cooc = pickle.load(f)
  return cooc