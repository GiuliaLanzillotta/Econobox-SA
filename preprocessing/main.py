import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from preprocessing.tokenizer import build_vocab
from preprocessing.lemmatizer import TxtLemmatized
from preprocessing.pipeline import run_preprocessing, getTxtLemmatization

if __name__ == "__main__":
    """
    build_vocab(frequency_treshold=0,
                file_name="full_vocab_in_stanford.pkl",
                input_files=["../data/twitter-datasets/replaced_train_neg_full.txt",
                             "../data/twitter-datasets/replaced_train_pos_full.txt"],
                use_base_vocabulary=True,
                base_vocabulary_name="stanford_vocab.pkl")
    
    getTxtLemmatization(input_files=["test_data_no_numbers.txt"],
                        stopwords=False,
                        replace=True,
                        lemmatize=False,
                        replace_stanford=True)
    """
    neg_cooc_params={'window_size':10, 'weighting':None, 'num_samples':5},  

    vocab, neg_cooc = run_preprocessing(vocab_name="vocab.pkl",
                      cooc_name="cooc.pkl",
                      neg_cooc_name="neg_cooc.pkl",  
                      to_build_vocab=False,
                      to_build_cooc=False,
                      use_neg_sampling=True,  
                      to_lemmatize_input=False, #if True we would lemmatize with stopwords = 0 and replacement = 1
                      vocab_build_params={},
                      cooc_build_params={},
                      neg_cooc__params=neg_cooc_params,  
                      input_files=None)    

    exit(0)
