# Here we should implement all the necessary tools to build a sentence embedding
# TODO: incorporate possibility of using 2 different embeddings
from . import matrix_train_location, vocab_location
from preprocessing import train_positive_location, train_negative_location, \
  train_positive_sample_location, train_negative_sample_location
from preprocessing.tokenizer import  tokenize_text
import numpy as np
import pickle


def build_sentence_embedding(sentence, emb_matrix, vocab):
    """
    Computes the embedding for a sentence given the single words 
    embeddings. 
    @:param sentence: list(str)
        A sentence is represented as a list of words.
    @:param emb_matrix: np.ndarray
        The trained embedding matrix.
    @:param vocab: dict
        The vocabulary in use.
    @:return np.array
        The embedding for the sentence.
    """
    #TODO: something more sophisticated here 
    emb_dim = emb_matrix.shape[1]
    sentence_emb = np.zeros(emb_dim)
    for word in sentence: 
        sentence_emb += emb_matrix[vocab.get(word)]
    return sentence_emb

def build_embedding_matrix(label, N, D, emb, output_location=matrix_train_location,
                           vocab_location = vocab_location):
    """
    Scans the positive and negative files and builds a matrix that will contain the 
    representtion of each tweet along with its label (1 for positive, 0 for negative.
    Parameters:
    ----------
    @:param label : bool
        Whether to include the labels in the output matrix.
        If labels is false, the output matrix will not contain the labels column
        (to use in test pipeline).
    @:param N : int
        The total number of tweets (#positive tweets + #negative tweets).
    @:param D : int
        The embedding dimension.
    @:param emb : np.ndarray
        The embedding matrix
    @:param output_location : str
        The path to the file where the final embedding will be stored.
    @:param vocab_location : str
        The path to the file where the vocavulary to use is stored.
    Returns
    --------
    @:return np.ndarray
        The output matrix containing the tweets embedding + labels if present
    """
    # initialization
    if label: out_dim1 = D + 1
    else:out_dim1 = D
    output = np.zeros((N, out_dim1))
    counter = 0
    # load vocabulary
    with open(vocab_location, 'rb') as f:
        vocab = pickle.load(f)
    for i, file in enumerate([train_negative_sample_location,
                              train_positive_sample_location]):
        label_value = i
        with open(file) as f:
            print("Working on ", file)
            # look at each tweet
            for l, line in enumerate(f):
                # Here we filter out the words that are not in the vocabulary
                sentence = tokenize_text(line)
                sentence_filtered = [t for t in sentence if t in vocab.keys()]
                # Get the tweet embedding
                sentence_emb = build_sentence_embedding(sentence_filtered, emb_matrix=emb,
                                                        vocab=vocab).reshape(1, -1)  # making sure it is a row vector
                # Save the tweet in the output matrix
                #TODO: make sure the order doesn't matter
                if not label:output[counter, :] = sentence_emb
                else:output[counter, :] = np.column_stack((sentence_emb, label_value))
                if l % 10000 == 0:
                    print(l)
                counter += 1
    # Saving the output
    np.savez(output_location, output)
    return output



