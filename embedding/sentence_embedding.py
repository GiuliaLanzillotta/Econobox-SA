# Here we should implement all the necessary tools to build a sentence embedding
# TODO: incorporate possibility of using 2 different embeddings
from embedding.embedding_base import EmbeddingBase
from preprocessing.tokenizer import tokenize_text
import numpy as np
#from flair.embeddings import TransformerDocumentEmbeddings
#from flair.data import Sentence

"""
NOTA BENE: every embedding function to use in the pipeline method as *aggregation_fun*
should take as input the *sentence* and the *embedding* (in this order) and should *tokenize*
the text, *filter* the sentence and finally *aggregate* the embeddings. 
"""

# global variable used to count the number
# of cut-offs when using the no embeddings sentence embedding
count = 0
frequency = 0

def no_embeddings(sentence, embedding, **kwargs):
    """
    Simply computes the mapping from word to vocabulary index
    for each word in the sentence and returns the index-version of
    the sentence.
    Note: this zero-embedding is useful when integrating the Embedding stage
    directly into the model (e.g. recurrentNN)
    :param sentence: list(str)
        A sentence is represented as a list of words.
    :param embedding: implements EmbeddingBase class
    :param kwargs:
        Supported additional arguments:
        - max_len (int) :
            Specifies the dimension of each sentence. Shorter sentences
            will be padded with 0, longer sentences will be cut off.
            The default is 100.
    :return: np.array
        The embedding for the sentence.
    """
    assert issubclass(embedding.__class__, EmbeddingBase), "embedding should be an instance of EmbeddingBase"
    vocabulary = embedding.vocabulary
    max_len = kwargs.get("max_len",100)
    # Here we filter out the words that are not in the vocabulary
    sentence = tokenize_text(sentence)
    sentence_filtered = [t for t in sentence if t in vocabulary.keys()]
    global frequency
    frequency += len(sentence_filtered)/len(sentence)
    sentence_emb = np.zeros(max_len)
    for i,word in enumerate(sentence_filtered):
        if i >= max_len:
            global count
            count += 1
            break # we're not going to add any more words to the sentence --> cutting it off
        sentence_emb[i] = vocabulary.get(word) + 1
    return sentence_emb

def sum_embeddings(sentence, embedding, weights=None, **kwargs):
    """
    Computes the embedding for a sentence given the single words
    embeddings by summing.
    @:param sentence: list(str)
        A sentence is represented as a list of words.
    @:param weights: a vectort with lenght equal to the sentence length.
        If weights are given than a weighted sum will be performed.
    @:param embedding: implements EmbeddingBase class
    @:return np.array
        The embedding for the sentence.
    """
    assert issubclass(embedding.__class__, EmbeddingBase), "embedding should be an instance of EmbeddingBase"
    embedding_matrix, vocabulary = embedding.embedding_matrix, embedding.vocabulary

    # Here we filter out the words that are not in the vocabulary
    sentence = tokenize_text(sentence)
    sentence_filtered = [t for t in sentence if t in vocabulary.keys()]
    # assigning equal weight to each word if the weights are not given
    if weights is None: weights=[1]*len(sentence_filtered)
    else: assert len(weights)==len(sentence_filtered), "The weights should have the same length as the sentence"
    # simply summing the words in the sentence
    emb_dim = embedding_matrix.shape[1]
    sentence_emb = np.zeros(emb_dim)
    for i,word in enumerate(sentence_filtered):
        sentence_emb += embedding_matrix[vocabulary.get(word) + 1]*weights[i]
    return sentence_emb

#def embedize(sentence, embedding):
#    tweet = Sentence(sentence)
#    embedding = TransformerDocumentEmbeddings(embedding)
#    embedding.embed(tweet)
#    tweet_emb = tweet.get_embedding()
#    tweet_emb_np = tweet_emb.cpu().detach().numpy()
#    return(tweet_emb_np)


