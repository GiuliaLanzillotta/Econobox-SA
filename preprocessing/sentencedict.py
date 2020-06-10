from data import sample_dimension, \
    train_negative_sample_location, train_positive_sample_location, test_location
import pickle

def get_sentence_dict(input_files):
  sentencedict = dict()
  for i, file in enumerate(input_files):

    with open(file) as f:
      for l, line in enumerate(f):
          sentencedict[line.strip()] = i

    with open('sentencedict.pkl', 'wb') as f:
      pickle.dump(sentencedict, f, pickle.HIGHEST_PROTOCOL)





get_sentence_dict([train_positive_sample_location, train_negative_sample_location])