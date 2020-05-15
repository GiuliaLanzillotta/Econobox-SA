# Here we should implement a vanilla NN that takes as input the embedding for 
# a sentence and prints as output the class of the tweet
from classifier.classifier_base import ClassifierBase
from matplotlib import pyplot as plt
from classifier import models_store_path
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import os


class vanilla_NN(ClassifierBase):
    """Vanilla NN classifier"""
    def __init__(self,
                 embedding_dimension,
                 name="VanillaNN"):
        super().__init__(embedding_dimension, name=name)
        self.history = None
        self.model = None


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


        self.model = Sequential()
        self.model.add(Dense(units=64, activation=activation,
                             input_dim=self.input_dim, name="Dense1"))
        self.model.add(Dense(units=64, activation=activation,
                             input_dim=64, name="Dense2"))
        self.model.add(Dense(units=2, activation='softmax', name="Dense3"))
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

        y_train = to_categorical(y)
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

        y_test = to_categorical(y)
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
        path = models_store_path+"/"+self.name
        os.makedirs(path)
        self.model.load_weights(path)



