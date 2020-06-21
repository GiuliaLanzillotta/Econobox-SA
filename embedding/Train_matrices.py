from flair.embeddings import TransformerDocumentEmbeddings
from preprocessing.tweetDF import load_tweetDF
from flair.data import Sentence
import numpy as np
from embedding import roberta_matrix_train_location


class Train_matrices():
    """
    Generating a train matrix by using a pretrained embedding to embed an entire tweet
    Parameters:
        embedding: the type of embedding used as a string, the different embeddings can be found here
            https://huggingface.co/transformers/pretrained_models.html
            example: "roberta-base"
        dim: the dimension of the embedding + 1 for the label
             For example the embedding dimension of BERT and roBERTa is 768
        data: the twitter data in a pandas dataframe format
        chunk_size: the amount of lines we look at at the same time
        output_location: where we store the train matrix. Note that if you create a new train matrix you should
        to define the location in embedding.__init__ and import the location to this file

    """
    def __init__(self, embedding, dim, data, chunk_size, output_location):
        self.embedding = embedding
        self.dim = dim
        self.data = data
        self.chunk_size = chunk_size
        self.output_location = output_location


    def embedize(self, data_subset_list):
        tweet = Sentence(data_subset_list)
        embedding = TransformerDocumentEmbeddings(self.embedding)
        embedding.embed(tweet)
        tweet_emb = tweet.get_embedding()
        tweet_emb_np = tweet_emb.detach().numpy()
        return(tweet_emb_np)


    def Train_matrices(self):
        #to be able to append, this first row will be deleted later
        output = np.zeros((1, self.dim))
        for start in range(0, self.data.shape[0], self.chunk_size):
            print(start)
            data_subset = self.data.iloc[start:start + self.chunk_size]
            data_subset_list = data_subset['text'].to_list()
            embeddinglist = list(map(self.embedize, data_subset_list ))
            data_matrix = np.append(embeddinglist, data_subset['sent'].values.reshape(-1,1), axis=1)
            output = np.vstack((output, data_matrix))

        output = output[0:-1,:]
        np.savez(self.output_location, output)
        return output





