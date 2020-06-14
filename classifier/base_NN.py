"""
Implements the base_NN abstract class, which is an instance of
BaseClassifier.
"""
from classifier.classifier_base import ClassifierBase
from matplotlib import pyplot as plt
from classifier import models_store_path, predictions_folder
from embedding.pipeline import generate_training_matrix, get_validation_data
from embedding import sentence_embedding
from abc import abstractmethod
import tensorflow as tf
import numpy as np
import math
import os

class BaseNN(ClassifierBase):

    def __init__(self,
                 embedding_dimension,
                 name,
                 embedding_matrix=None,# initializer of embedding
                 vocabulary_dimension=200):
        super().__init__(embedding_dimension, name=name)
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

    @abstractmethod
    def build(self, **kwargs):
        """Build the net here."""
        pass

    def train(self, x, y, **kwargs):
        """
        Training the model and saving the history of training.
        """
        print("Training model.")
        generator_mode = kwargs.get("generator_mode")
        epochs = kwargs.get("epochs",10)
        batch_size = kwargs.get("batch_size",32)
        validation_split = kwargs.get("validation_split",0.2)
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                              min_delta=0.0001,
                                                              patience=1)
        if not generator_mode:
            y_train = tf.keras.utils.to_categorical(y)
            self.history = self.model.fit(x, y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=validation_split,
                                      callbacks=[earlystop_callback],
                                      shuffle=True)
        else:
            # extracting relevant arguments
            embedding = kwargs.get("embedding")
            input_files = kwargs.get("input_files")
            label_values = kwargs.get("label_values")
            input_entries = kwargs.get("input_entries")
            max_len = kwargs.get("max_len")
            n_steps = int((1-validation_split)*input_entries/batch_size)
            validation_data = get_validation_data(embedding=embedding,
                                                  input_files=input_files,
                                                  label_values=label_values,
                                                  validation_split=validation_split,
                                                  categorical=True,
                                                  aggregation_fun=sentence_embedding.no_embeddings,
                                                  input_entries=input_entries,
                                                  sentence_dimesion=max_len)
            self.history = self.model.fit(generate_training_matrix(embedding=embedding,
                                                                   input_files=input_files,
                                                                   label_values=label_values,
                                                                   chunksize=batch_size,
                                                                   validation_split=validation_split,
                                                                   categorical=True,
                                                                   aggregation_fun=sentence_embedding.no_embeddings,
                                                                   input_entries=input_entries,
                                                                   sentence_dimesion=max_len),
                                          callbacks=[earlystop_callback],
                                          epochs=epochs, steps_per_epoch =n_steps,
                                          validation_data=validation_data)
        self.plot_history()

    def make_predictions(self, x, save=True, **kwargs):
        print("Making predictions")
        preds = self.model.predict(x)
        preds_classes = np.argmax(preds, axis=-1).astype("int")
        preds_classes[preds_classes == 0] = -1
        if save: self.save_predictions(preds_classes)

    @staticmethod
    def analyse_worst_predictions(x, y, pred_probs, idx2word, n=10):
        """ Picking the n most wrong predictions and printing the
            sentences."""
        print("Analysing the mistakes of the model.")
        # extracting the mistakes over which we were more confident
        temp = np.column_stack([x,np.array(y).reshape(-1,1),pred_probs])
        mistakes = temp[:,x.shape[1]] != np.argmax(temp[:,-2:], axis=1)
        temp_ms = temp[mistakes]
        n_worst_positive_indices = np.argsort(temp_ms[:,-1])[-n:]
        n_worst_negative_indices = np.argsort(temp_ms[:,-2])[-n:]
        n_worst_positive = temp_ms[n_worst_positive_indices,:x.shape[1]]
        n_positive_confidences = temp_ms[n_worst_positive_indices,-1]
        n_worst_negative = temp_ms[n_worst_negative_indices,:x.shape[1]]
        n_negative_confidences = temp_ms[n_worst_negative_indices,-2]
        # turning to words
        to_words = lambda idx: idx2word.get(idx-1, "")
        #because the sentence embedding is created
        #sentence_emb[i] = vocabulary.get(word) + 1
        n_worst_negative_words = np.vectorize(to_words)(n_worst_negative)
        n_worst_negative_sentences = [" ".join(line) for line in n_worst_negative_words]
        n_worst_positive_words = np.vectorize(to_words)(n_worst_positive)
        n_worst_positive_sentences = [" ".join(line) for line in n_worst_positive_words]
        print("False negatives:")
        for i, sentence in enumerate(n_worst_negative_sentences):
            print(sentence)
            print("Confidence: ", n_negative_confidences[i])
        print("False positives:")
        for i, sentence in enumerate(n_worst_positive_sentences):
            print(sentence)
            print("Confidence: ", n_positive_confidences[i])
        return

    def test(self, x, y,**kwargs):
        print("Testing model")
        idx2word = kwargs.get("idx2word")

        #y_test = tf.keras.utils.to_categorical(y)
        prediction = self.model.predict(x, verbose=0)
        prediction_probs = tf.nn.softmax(prediction, axis=1).numpy()
        prediction_classes = np.argmax(prediction, axis=-1).astype("int")

        self.score_model(true_classes=y,
                         predicted_classes=prediction_classes,
                         predicted_probabilities=prediction_probs)

        self.analyse_worst_predictions(x,y,prediction_probs,idx2word,n=5)

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
        #todo: save the plot automatically

    def save(self, overwrite=True, **kwargs):
        print("Saving model")
        abs_path = os.path.abspath(os.path.dirname(__file__))
        path = models_store_path+self.name+"/"
        if not os.path.exists(os.path.join(abs_path,path)):
            os.makedirs(os.path.join(abs_path,path))
        self.model.save_weights(os.path.join(abs_path,path),overwrite=overwrite)

    def load(self, **kwargs):
        print("Loading model")
        path = models_store_path+self.name+"/"
        abs_path = os.path.abspath(os.path.dirname(__file__))
        self.model.load_weights(os.path.join(abs_path,path))