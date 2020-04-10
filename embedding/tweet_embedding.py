# Here we should implement all the necessary tools to build a sentence embedding 
import numpy as np

def build_sentence_embedding(sentence, emb_matrix, vocab):
    """
    Computes the embedding for a sentence given the single words 
    embeddings. 
    @param sentence: list(str)
        A sentence is represented as a list of words.
    @param emb_matrix: np.ndarray
        The trained embedding matrix.
    @param vocab: dict
        The vocabulary in use.
    @returns np.array
        The embedding for the sentence.
    """
    #TODO: something more sophisticated here 
    emb_dim = emb_matrix.shape[1]
    sentence_emb = np.zeros(emb_dim)
    for word in sentence: 
        sentence_emb += emb_matrix[vocab.get(word)]
    return sentence_emb