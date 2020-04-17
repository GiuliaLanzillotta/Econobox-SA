# Main of the embedding module
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from embedding.pipeline import run_embedding_pipeline


if __name__ == "__main__":
    run_embedding_pipeline()
    exit(0)