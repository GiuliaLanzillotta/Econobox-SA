from classifier.classifier_base import ClassifierBase
from embedding.BERT_EMBEDDING import get_segments, get_masks, get_ids
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
from classifier import models_store_path
import numpy as np
import os
import bert
FullTokenizer = bert.bert_tokenization.FullTokenizer
from keras.utils import to_categorical



class BERT_NN(ClassifierBase):
    """Neural net with Bert embedding layer"""

    def __init__(self,
                 max_seq_length,
                 embedding_dimension,
                 name="BERT_NN"):
        super().__init__(embedding_dimension, name=name)
        self.history = None
        self.model = None
        self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True)
        self.max_seq_length = max_seq_length


    def bert_preprocessing(self, x, **kwargs):
        bert_layer = self.bert_layer
        max_seq_length = self.max_seq_length
        ##Defining/creating the bert tokenizer
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        tokenizer = FullTokenizer(vocab_file, do_lower_case)
        ##tokenizing the input
        xtokens = tokenizer.tokenize(x)
        ##preprocessing the input further
        input_ids = get_ids(xtokens, tokenizer, max_seq_length)
        input_masks = get_masks(xtokens, max_seq_length)
        input_segments = get_segments(xtokens, max_seq_length)

        return([[input_ids], [input_masks], [input_segments]])



    def build(self, **kwargs):
        ##extracting model parameters from arguments
        activation = kwargs.get("activation", "relu")
        metrics = kwargs.get("metrics", ['accuracy'])
        optim = kwargs.get("optimizer", 'adam')
        ## CREATING THE GRAPH NODES and CONNECTING THE GRAPH
        max_seq_length = self.max_seq_length
        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
        bert_layer = self.bert_layer
        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
        dense1 = tf.keras.layers.Dense(64, activation=activation, name="Dense1")(pooled_output)
        dense2 = tf.keras.layers.Dense(64, activation=activation, name="Dense2")(dense1)
        dense3 = tf.keras.layers.Dense(2, activation='softmax', name='Dense3')(dense2)

        ## COMPILING THE MODEL
        self.model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=dense3)
        optimizer = optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics)
        self.model.summary()


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

        x_preprpcessed = self.bert_preprocessing(x)

        y_train = to_categorical(y)
        self.history = self.model.fit(x_preprpcessed, y_train,
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

    def make_predictions(self, x, save=True, **kwargs):
        print("Making predictions")
        preds = self.model.predict(x)
        preds_classes = np.argmax(preds, axis=-1).astype("int")
        preds_classes[preds_classes == 0] = -1
        if save: self.save_predictions(preds_classes)


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









