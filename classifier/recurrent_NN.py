# Here we should implement a recurrent NN that takes as input
# a sequence of embeddings (one for each word) and
# outputs the class of the tweet
from classifier.classifier_base import ClassifierBase
from matplotlib import pyplot as plt
from classifier import models_store_path
import tensorflow as tf


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
    - attention mechanism
    """
    def __init__(self,
                 embedding_dimension,
                 name="RecurrentNN",
                 cell_type="LSTM", # either LSTM or GRU
                 num_layers=1, # number of recurrent layers
                 hidden_size=64, # size of hidden representation
                 train_embedding=False, # whether to add an Embedding layer to the model
                 use_attention= False):  # whether to add an attention mechanism to the model
        super().__init__(embedding_dimension, name=name)
        possible_cell_values = ["GRU","LSTM"]
        self.history = None
        self.model = None
        assert cell_type in possible_cell_values, "The cell type must be one of the following values : " \
                                                  + " ".join(possible_cell_values) # printing the admissible cell values
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # TODO: incorporate embedding
        # TODO: incorporate attention mechanism


    def build(self, **kwargs):
        print("Building model.")

        activation = kwargs.get("activation")
        loss = kwargs.get("loss")
        optimizer = kwargs.get("optimizer")
        metrics = kwargs.get("metrics")

        if not activation: activation = "relu"
        if not loss: loss = "binary_crossentropy"
        if not optimizer: optimizer = "adam"
        if not metrics : metrics = ['accuracy']


        model = tf.keras.models.Sequential()
        ## Recurrent layer -------
        ## This part of the model is responsible for processing the sequence
        if self.cell_type == "GRU":
            recurrent = tf.keras.layers.GRU
        if self.cell_type == "LSTM":
            recurrent = tf.keras.layers.LSTM
        # Stacking num_layers recurrent layers
        for l in range(self.num_layers-1):
            # Note: forcing the recurrent layer to be Bidirectional
            model.add(tf.keras.layers.Bidirectional(recurrent(self.hidden_size), return_sequences=True,
                                                    name="Recurrent"+str(l)))
        model.add(tf.keras.layers.Bidirectional(recurrent, name="Recurrent"+str(self.num_layers-1)))
        ## Dense head --------
        ## This last part of the model is responsible for mapping
        ## back the output of the recurrent layer to a binary value,
        ## which will indeed be our prediction
        model.add(tf.keras.layers.Dense(self.hidden_size, activation=activation, name="Dense1"))
        model.add(tf.keras.layers.Dense(1, name="Dense2"))
        self.model = model
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics)
        print(self.model.summary())

    def train(self, x, y, **kwargs):
        """
        Training the model and saving the history of training.
        """
        print("Training model.")
        epochs = kwargs.get("epochs")
        batch_size = kwargs.get("batch_size")
        validation_split = kwargs.get("validation_split")

        if not epochs: epochs = 10
        if not batch_size: batch_size=32
        if not validation_split: validation_split=0.20

        y_train = tf.keras.utils.to_categorical(y)
        self.history = self.model.fit(x, y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=validation_split,
                                      shuffle=True)
        self.plot_history()

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
        path = models_store_path+self.name
        self.model.save_weights(path,overwrite=overwrite)

    def load(self, **kwargs):
        print("Loading model")
        path = models_store_path+self.name
        self.model.load_weights(path)



