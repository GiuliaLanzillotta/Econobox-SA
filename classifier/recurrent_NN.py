# Here we should implement a recurrent NN that takes as input
# a sequence of embeddings (one for each word) and
# outputs the class of the tweet
from classifier.classifier_base import ClassifierBase
from matplotlib import pyplot as plt
from classifier import models_store_path, predictions_folder
import tensorflow as tf
import numpy as np
import os

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

    @staticmethod
    def get_cell(cell_name):
        """
        Simply a routine to switch to the right recurrent cell
        :param cell_name: name of the cell to use
        """
        if cell_name == "GRU":
            return tf.keras.layers.GRU
        if cell_name == "LSTM":
            return tf.keras.layers.LSTM
        raise NotImplementedError("cell type "+cell_name+" not supported!")

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
        recurrent = self.get_cell(cell_type)
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

#TODO: add penalty term for attention matrix
class attention_NN(recurrent_NN):
    """Extends the recurrent model with self-attention"""
    def __init__(self,
                 embedding_dimension,
                 vocabulary_dimension,
                 embedding_matrix=None,
                 name="AttentionNN"):
        super().__init__(embedding_dimension, vocabulary_dimension, embedding_matrix, name)

    def build(self, **kwargs):
        """Builds the attention NN computational graph"""
        print("Building model.")
        ## --------------------
        ## Extracting model parameters from arguments
        cell_type = kwargs.get("cell_type")  # either LSTM or GRU
        assert cell_type in self.possible_cell_values, "The cell type must be one of the following values : " \
                                                       + " ".join(
            self.possible_cell_values)  # printing the admissible cell values
        hidden_size = kwargs.get("hidden_size", 64)  # size of hidden representation
        train_embedding = kwargs.get("train_embedding", False)  # whether to train the Embedding layer to the model
        use_pretrained_embedding = kwargs.get("use_pretrained_embedding", False)
        dropout_rate = kwargs.get("dropout_rate", 0.0)  # setting the dropout to 0 is equivalent to not using it
        use_normalization = kwargs.get("use_normalization", False)
        activation = kwargs.get("activation", "relu")
        metrics = kwargs.get("metrics", ['accuracy'])
        optim = kwargs.get("optimizer", "sgd")
        att_key_dim = kwargs.get("att_key_dim", 64)

        ## ---------------------
        ## CREATING THE GRAPH NODES
        inputs = tf.keras.layers.Input(shape=(None,), dtype='int32')
        weights = None
        if use_pretrained_embedding: weights = [self.embedding_matrix]
        embedding = tf.keras.layers.Embedding(input_dim=self.vocabulary_dim,
                                              output_dim=self.input_dim,
                                              weights=weights,
                                              mask_zero=True,
                                              trainable=train_embedding)
        masking = tf.keras.layers.Masking(mask_value=0)
        recurrent_cell = self.get_cell(cell_type)
        recurrent_layer = tf.keras.layers.Bidirectional(recurrent_cell(hidden_size,
                                                                       return_sequences=True,
                                                                       recurrent_dropout=dropout_rate),
                                                        merge_mode="concat",
                                                        name="RecurrentLayer")
        # the output of the recurrent layer should have dimension [batch_size, timesteps, hidden_size*2]
        # since we're merging the hidden states with concatenation
        dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        # SELF ATTENTION
        query = tf.keras.layers.Dense(att_key_dim, activation="tanh", name="attention1")
        score = tf.keras.layers.Dense(1, name="attention2")
        # DENSE head
        dense1 = tf.keras.layers.Dense(hidden_size, activation=activation, name="Dense1")
        if use_normalization: norm = tf.keras.layers.BatchNormalization()
        dense2 = tf.keras.layers.Dense(2, name="Dense2")
        ## CONNECTING THE GRAPH
        embedded_inputs = embedding(inputs)
        masked_inputs = masking(embedded_inputs)
        recurrent_output = recurrent_layer(masked_inputs) # tensor with all hidden states
        queries = query(recurrent_output)
        scores = score(queries) # batch x timesteps x 1
        weights = tf.nn.softmax(scores, axis=1) # batch x timesteps x 1
        context_vector = weights * recurrent_output # should be element-wise mult.
        context_vector = tf.reduce_sum(context_vector, axis=1) # summing the timesteps
        dropped_out = dropout(context_vector)
        if use_normalization: dropped_out = norm(dropped_out)
        dense1_out = dense1(dropped_out)
        outputs = dense2(dense1_out)
        ## COMPILING THE MODEL
        self.model = tf.keras.Model(inputs = inputs, outputs=outputs)
        optimizer = self.get_optimizer(optim)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics)
        print(self.model.summary())

    def get_attention_weights(self, sentence):
        """ Helper function for interpretability of the model."""
        sentence_with_batch_size = tf.expand_dims(sentence, axis=0)
        ## forward passing through the model
        embedding = self.model.get_layer("embedding")
        embedded_inputs = embedding(sentence_with_batch_size)
        masking = self.model.get_layer("masking")
        masked_inputs = masking(embedded_inputs)
        recurrent_layer = self.model.get_layer("RecurrentLayer")
        recurrent_output = recurrent_layer(masked_inputs)  # tensor with all hidden states
        query = self.model.get_layer("attention1")
        queries = query(recurrent_output)
        score = self.model.get_layer("attention2")
        scores = score(queries)
        weights = tf.nn.softmax(scores, axis=1)
        return weights

    def visualize_attention(self, sentence):
        """ Helper function for interpretability of the model.
        :param sentence: (str) example sentence to use.
        """
        sequence_len = len(sentence)
        weights = self.get_attention_weights(sentence)
        attention_plot = np.zeros((sequence_len, sequence_len))
        attention_weights = tf.reshape(weights, (-1,sequence_len)) # 1 x timesteps
        #TODO: follow this model for visualization
        # https://github.com/kionkim/stat_analysis/blob/master/notebooks/text_classification_RNN_SA_umich.ipynb

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(weights, cmap='viridis')

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + sentence, fontdict=fontdict)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
