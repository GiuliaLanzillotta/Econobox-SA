# Here we should implement a vanilla NN that takes as input the embedding for 
# a sentence and prints as output the class of the tweet
from classifier.classifier_base import ClassifierBase
from matplotlib import pyplot as plt
import tensorflow as tf
from classifier import models_store_path
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense


class vanilla_NN(tf.keras.Model, ClassifierBase):
    """Vanilla NN classifier"""
    def __init__(self,
                 embedding_dimension,
                 name="VanillaNN"):
        super(vanilla_NN, self).__init__()
        super(ClassifierBase, self).__init__(embedding_dimension, name)
        self.history = None


    def build(self,
              activation='relu',
              optimizer='adam',
              loss="binary_crossentropy",
              metrics=None):
        print("Building model.")
        if metrics is None:
            metrics = ['accuracy']
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

    def train(self,
              x, y,
              epochs=10,
              batch_size=32,
              validation_split=0.2):
        """
        Training the model and saving the history of training.
        """
        print("Training model.")
        y_train = to_categorical(y)
        self.history = self.model.fit(x, y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=batch_size,
                                      shuffle=True)
        self.plot_history()

    def test(self, x, y,
             batch_size=32,
             verbose=1):
        print("Testing model")
        y_test = to_categorical(y)
        self.model.evaluate(x, y_test,
                            batch_size=batch_size,
                            verbose=verbose)

    def plot_history(self):
        #TODO:fix this method

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


