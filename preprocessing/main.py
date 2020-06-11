import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from preprocessing.lemmatizer import TxtLemmatized
from preprocessing.pipeline import run_preprocessing, getTxtLemmatization

if __name__ == "__main__":
    getTxtLemmatization(input_files=["test_data.txt"],
                        stopwords=False,
                        replace=True,
                        lemmatize=False,
                        replace_stanford=True)
    exit(0)