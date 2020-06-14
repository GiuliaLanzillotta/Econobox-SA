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
    """
    getTxtLemmatization(input_files=["test_data_no_numbers.txt"],
                        stopwords=False,
                        replace=True,
                        lemmatize=False,
                        replace_stanford=True)

    exit(0)