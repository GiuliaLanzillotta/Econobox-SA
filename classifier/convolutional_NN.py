# Here is the code for a convolutional NN that takes as input a
# sequence of token-ids (one for each word) and outputs the
# class prediction for the tweet
from classifier.classifier_base import ClassifierBase
from matplotlib import pyplot as plt
from classifier import models_store_path, predictions_folder
from embedding.pipeline import generate_training_matrix, \
    get_validation_data
from matplotlib import pyplot as plt
from classifier.base_NN import BaseNN
import tensorflow as tf
import seaborn as sb
import numpy as np
import math

class convolutional_NN(BaseNN):
    """
    Convolutional NN classifier
    This class wraps a classifier based on 1D convolutions.
    ---------
    """

    def __init__(self,
                 embedding_dimension,
                 vocabulary_dimension,
                 embedding_matrix=None, # initializer of embedding
                 name="ConvolutionalNN"):
        super().__init__(embedding_dimension=embedding_dimension,
                         vocabulary_dimension=vocabulary_dimension,
                         name=name,
                         embedding_matrix=embedding_matrix)

    @staticmethod
    def get_pooling_layer(pooling_type, pool_size=2):
        """ Switching method for pooling layer."""
        if pooling_type=="max":
            return tf.keras.layers.MaxPool1D(pool_size=pool_size)
        if pooling_type=="avg":
            return tf.keras.layers.AvgPool1D(pool_size=pool_size)

    def build(self, **kwargs):
        print("Building model.")
        ## -------------------
        ## EXTRACTING ARGUMENTS
        train_embedding = kwargs.get("train_embedding", False)  # whether to train the Embedding layer to the model
        use_pretrained_embedding = kwargs.get("use_pretrained_embedding", True)
        pooling = kwargs.get("use_pooling",True)
        pooling_type = kwargs.get("pooling_type","max")
        num_convolutions = kwargs.get("num_convolutions",10)
        window_size = kwargs.get("window_size",5)
        dilation_rate = kwargs.get("dilation_rate",1)
        pool_size = kwargs.get("pool_size",2)
        hidden_size = kwargs.get("hidden_size", 64)  # size of hidden representation
        dropout_rate = kwargs.get("dropout_rate", 0.0)  # setting the dropout to 0 is equivalent to not using it
        use_normalization = kwargs.get("use_normalization", False)
        activation = kwargs.get("activation", "relu")
        metrics = kwargs.get("metrics", ['accuracy'])
        optim = kwargs.get("optimizer", "sgd")
        ## -------------------
        ## CREATING THE NODES
        inputs = tf.keras.layers.Input(shape=(None,), dtype='int32')
        weights = None
        if use_pretrained_embedding: weights = [self.embedding_matrix]
        embedding = tf.keras.layers.Embedding(input_dim=self.vocabulary_dim,
                                              output_dim=self.input_dim,
                                              weights=weights,
                                              mask_zero=True,
                                              trainable=train_embedding)
        masking = tf.keras.layers.Masking(mask_value=0)
        convolutions = []
        channels = self.input_dim
        for l in range(num_convolutions):
            if l > num_convolutions//2: channels = channels * 2
            layer = tf.keras.layers.Conv1D(filters=channels,
                                           kernel_size=window_size,
                                           strides=1,
                                           padding="same",
                                           dilation_rate=dilation_rate,
                                           activation="relu",
                                           name='convolution{}'.format(str(l + 1)))
            convolutions.append(layer)
        flattening = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")
        if pooling: pool = self.get_pooling_layer(pooling_type, pool_size)
        # DENSE head
        dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        dense1 = tf.keras.layers.Dense(hidden_size, activation=activation, name="Dense1")
        if use_normalization: norm = tf.keras.layers.BatchNormalization()
        dense2 = tf.keras.layers.Dense(2, name="Dense2")
        ## -------------------
        ## CONNECTING THE NODES
        embedded_inputs = embedding(inputs)
        masked_inputs = masking(embedded_inputs)
        conv_input = masked_inputs
        for conv_layer in convolutions:
            conv_res = conv_layer(conv_input)
            if pooling: conv_res = pool(conv_res)
            conv_input = conv_res
        if use_normalization: norm(conv_res)
        flattened = flattening(conv_res)
        flattened = dropout(flattened)
        dense1_out = dense1(flattened)
        outputs = dense2(dense1_out)
        ## -------------------
        ## COMPILING THE MODEL
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = self.get_optimizer(optim)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics,
                           experimental_run_tf_function=False)
        self.model.summary()



