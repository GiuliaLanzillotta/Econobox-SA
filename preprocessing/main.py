import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from preprocessing.pipeline import run_preprocessing

if __name__ == "__main__":
    vocab, cooc = run_preprocessing(vocab_name="vocab.pkl",
                                   cooc_name="cooc.pkl",
                                   to_build_vocab=False,
                                   to_build_cooc=False)
    exit(0)