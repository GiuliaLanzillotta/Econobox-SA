# Here we should implement a vanilla NN that takes as input the embedding for 
# a sentence and prints as output the class of the tweet
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense


class vanilla_NN(tf.keras.Model):
    """Vanilla NN classifier"""
    def __init__(self, emb_dim, LEARNING_RATE = 0.01,
                 MOMENTUM = 0.9 ,BATCH_SIZE = 32,
                 EPOCHS = 5 , VAL_SPLIT = 0.3):
        super(vanilla_NN, self).__init__()
        self.input_dim = emb_dim
        self.LEARNING_RATE = LEARNING_RATE
        self.MOMENTUM = MOMENTUM
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.VAL_SPLIT = VAL_SPLIT
        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu', input_dim=self.input_dim, name="Dense1"))
        self.model.add(Dense(units=64, activation='relu', input_dim=64, name="Dense2"))
        self.model.add(Dense(units=2, activation='softmax', name="Dense3"))

    def compile_model(self, optimizer='adam',
                      loss="binary_crossentropy",
                      metrics=['accuracy']):
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics)
        print(self.model.summary())

    def fit_model(self, x=None, y=None):
        y_train = to_categorical(y)
        self.history = self.model.fit(x, y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                            validation_split=self.BATCH_SIZE, shuffle=True)

    def plot_history(self):
        #TODO:fix this method

        # summarize history for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
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

    # TODO : add method to save model

