from tokenizer import build_vocab
from cooc import build_cooc

if __name__ == "__main__":
    print("Building vocabulary")
    build_vocab(frequency_treshold=10)
    print("Building cooc")
    build_cooc(vocab_location="../vocab.pkl")