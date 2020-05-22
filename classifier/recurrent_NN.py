# Here we should implement a recurrent NN that takes as input
# a sequence of embeddings (one for each word) and
# outputs the class of the tweet
from classifier.classifier_base import ClassifierBase
from matplotlib import pyplot as plt
from classifier import models_store_path, predictions_folder
import tensorflow as tf
import numpy as np
import os


class BahdanauAttention(tf.keras.layers.Layer):
    """Implements an attention layer as described by Bahdanau."""
    def __init__(self,
                 units,
                 **kwargs):
        super().__init__(**kwargs)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query=None, values=None):
        """
        :param query: The embedded word on which we should query.
                shape == (batch_size, hidden size)
        :param values: The sequence of words which constitutes the sentence.
                shape == (batch_size, max_len, hidden size)
        :return: context vector: (batch_size, hidden_size) tensor,
                attn weights: (batch_size, max_length, 1) tensor
            The context vector is a weighted sum of the input sequence that puts
            more weight on the elements more similar to the given query.

        """
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class AttentionWrapper(tf.keras.layers.Layer):
    """ Uses attention with 2 queries, given the output
        of a recurrent layer."""
    def __init__(self,
                 vocab2index,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab2index = vocab2index
        self.attention = BahdanauAttention(64)

    def __call__(self, sequence, *args, **kwargs):
        #TODO: call attention for a positive query
        #TODO: call attention for a negative query
        #TODO: dense connect the two outputs
        pass


#TODO: connect attention to the recurrent net
class recurrent_NN(ClassifierBase):
    """
    Recurrent NN classifier
    This class implements a recurrent NN.
    -------------
    Parameters :
    - cell type (LSTM, GRU, ...)
    - number of recurrent layers to use
    - size of hidden representation
    ------------
    Addons:
    - embedding layer
    ------------

    """


    def __init__(self,
                 embedding_dimension,
                 vocabulary_dimension,
                 embedding_matrix=None, # initializer of embedding
                 name="RecurrentNN"):
        super().__init__(embedding_dimension, name=name)
        self.possible_cell_values = ["GRU","LSTM"]
        self.history = None
        self.model = None
        self.embedding_matrix = embedding_matrix
        self.vocabulary_dim = vocabulary_dimension + 1

    @staticmethod
    def get_optimizer(optim_name):
        """
        Simply a routine to switch to the right optimizer
        :param optim_name: name of the optimizer to use
        """
        learning_rate = 0.001
        momentum = 0.09
        optimizer = None
        if optim_name == "sgd": optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        if optim_name == "adam": optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if optim_name == "rmsprop": optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,
                                                                            momentum=momentum)
        return optimizer


    def build(self, **kwargs):
        #TODO: add support for list of hidden sizes
        print("Building model.")

        ## --------------------
        ## Extracting model parameters from arguments
        cell_type = kwargs.get("cell_type")  # either LSTM or GRU
        assert cell_type in self.possible_cell_values, "The cell type must be one of the following values : " \
                                                       + " ".join(self.possible_cell_values)  # printing the admissible cell values
        num_layers = kwargs.get("num_layers",1)  # number of recurrent layers
        hidden_size = kwargs.get("hidden_size",64)  # size of hidden representation
        train_embedding = kwargs.get("train_embedding",False)  # whether to train the Embedding layer to the model
        use_pretrained_embedding = kwargs.get("use_pretrained_embedding",False)
        dropout_rate = kwargs.get("dropout_rate",0.0) # setting the dropout to 0 is equivalent to not using it
        use_normalization = kwargs.get("use_normalization",False)
        activation = kwargs.get("activation","relu")
        metrics = kwargs.get("metrics",['accuracy'])
        optim = kwargs.get("optimizer","sgd")
        ## ---------------------


        model = tf.keras.models.Sequential()
        # Note the shape parameter must not include the batch size
        # Here None stands for the timesteps
        model.add(tf.keras.layers.Input(shape=(None,), dtype='int32'))
        weights = None
        if use_pretrained_embedding: weights = [self.embedding_matrix]
        model.add(tf.keras.layers.Embedding(input_dim=self.vocabulary_dim,
                                            output_dim=self.input_dim,
                                            weights=weights,
                                            mask_zero=True,
                                            trainable=train_embedding))
        model.add(tf.keras.layers.Masking(mask_value=0))
        ## Recurrent layer -------
        ## This part of the model is responsible for processing the sequence
        if cell_type == "GRU":
            recurrent = tf.keras.layers.GRU
        if cell_type == "LSTM":
            recurrent = tf.keras.layers.LSTM
        # Stacking num_layers recurrent layers
        for l in range(num_layers-1):
            # Note: forcing the recurrent layer to be Bidirectional
            model.add(tf.keras.layers.Bidirectional(recurrent(hidden_size,
                                                              return_sequences=True,
                                                              recurrent_dropout=dropout_rate),
                                                    name="Recurrent"+str(l)))
        model.add(tf.keras.layers.Bidirectional(recurrent(hidden_size),
                                                name="Recurrent"+str(num_layers-1)))
        ## Dense head --------
        ## This last part of the model is responsible for mapping
        ## back the output of the recurrent layer to a binary value,
        ## which will indeed be our prediction
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        model.add(tf.keras.layers.Dense(hidden_size, activation=activation, name="Dense1"))
        if use_normalization: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(2, name="Dense2"))
        self.model = model
        optimizer = self.get_optimizer(optim)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics)
        print(self.model.summary())

    def train(self, x, y, **kwargs):
        """
        Training the model and saving the history of training.
        """
        print("Training model.")
        epochs = kwargs.get("epochs",10)
        batch_size = kwargs.get("batch_size",32)
        validation_split = kwargs.get("validation_split",0.2)

        y_train = tf.keras.utils.to_categorical(y)
        self.history = self.model.fit(x, y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=validation_split,
                                      shuffle=True)
        self.plot_history()

    def make_predictions(self, x, save=True, **kwargs):
        print("Making predictions")
        preds = self.model.predict(x)
        preds_classes = np.argmax(preds, axis=-1).astype("int")
        preds_classes[preds_classes == 0] = -1
        if save: self.save_predictions(preds_classes)


    def test(self, x, y,**kwargs):
        print("Testing model")
        batch_size = kwargs.get("batch_size")
        verbose = kwargs.get("verbose")
        if not batch_size: batch_size = 32
        if not verbose: verbose=1

        y_test = tf.keras.utils.to_categorical(y)
        self.model.evaluate(x, y_test,
                            batch_size=batch_size,
                            verbose=verbose)

    def plot_history(self):
        # summarize history for accuracy
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def save(self, overwrite=True, **kwargs):
        print("Saving model")
        path = models_store_path+self.name+"/"
        try: os.makedirs(path)
        except Exception as e: print(e)
        self.model.save_weights(path,overwrite=overwrite)

    def load(self, **kwargs):
        print("Loading model")
        path = models_store_path+self.name+"/"
        self.model.load_weights(path)

