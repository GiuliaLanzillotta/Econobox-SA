# Here we should implement all the necessary tools to build a sentence embedding
# TODO: incorporate possibility of using 2 different embeddings
from embedding.embedding_base import EmbeddingBase
from preprocessing.tokenizer import tokenize_text
import numpy as np

"""
NOTA BENE: every embedding function to use in the pipeline method as *aggregation_fun*
should take as input the *sentence* and the *embedding* (in this order) and should *tokenize*
the text, *filter* the sentence and finally *aggregate* the embeddings. 
"""
def no_embeddings(sentence, embedding):
    """
    Simply computes the mapping from word to vocabulary index
    for each word in the sentence and returns the index-version of
    the sentence
    :param sentence: list(str)
        A sentence is represented as a list of words.
    :param embedding: implements EmbeddingBase class
    :return: np.array
        The embedding for the sentence.
    """
    assert issubclass(embedding.__class__, EmbeddingBase), "embedding should be an instance of EmbeddingBase"
    vocabulary = embedding.vocabulary

    # Here we filter out the words that are not in the vocabulary
    sentence = tokenize_text(sentence)
    sentence_filtered = [t for t in sentence if t in vocabulary.keys()]
    sentence_emb = np.zeros(len(sentence))
    for i,word in enumerate(sentence_filtered):
        sentence_emb[i] = vocabulary.get(word)
    return sentence_emb

def sum_embeddings(sentence, embedding):
    """
    Computes the embedding for a sentence given the single words
    embeddings by summing.
    @:param sentence: list(str)
        A sentence is represented as a list of words.
    @:param embedding: implements EmbeddingBase class
    @:return np.array
        The embedding for the sentence.
    """
    assert issubclass(embedding.__class__, EmbeddingBase), "embedding should be an instance of EmbeddingBase"
    embedding_matrix, vocabulary = embedding.embedding_matrix, embedding.vocabulary


    # Here we filter out the words that are not in the vocabulary
    sentence = tokenize_text(sentence)
    sentence_filtered = [t for t in sentence if t in vocabulary.keys()]

    # simply summing the words in the sentence
    emb_dim = embedding_matrix.shape[1]
    sentence_emb = np.zeros(emb_dim)
    for word in sentence_filtered:
        sentence_emb += embedding_matrix[vocabulary.get(word)]
    return sentence_emb


