# Here we should implement a recurrent NN that takes as input
# a sequence of embeddings (one for each word) and
# outputs the class of the tweet
from matplotlib import pyplot as plt
from classifier.base_NN import BaseNN
import tensorflow as tf
import seaborn as sb
import numpy as np
import math

class recurrent_NN(BaseNN):
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
        super().__init__(embedding_dimension=embedding_dimension,
                         vocabulary_dimension=vocabulary_dimension,
                         name=name,
                         embedding_matrix=embedding_matrix)
        self.possible_cell_values = ["GRU","LSTM"]

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

    @staticmethod
    def build_convolutions(dilation_rate,
                           initial_filter_size,
                           window_size,
                           input_length,
                           threshold_channels=600,
                           num_layers=None):
        """Builds the convolution segment in the network.
            :param threshold_channels: maximum number of channels
            :param num_layers: number of convolutions to perform
            :param dilation_rate: (float) dilation rate to use in the convolution
            :param initial_filter_size: (int) first convolution layer number of output filters.
                    For each layer this number will be doubled.
            :param window_size: (int) the convolution layer window size.
            :param input_length: (int) length of the input
            -----------------
            About convolution:
            # CONVOLUTION LAYERS (to use instead of flattening)
            # we now have a [batch_size x heads x hidden_dim*2 ] vector, where each of the
            # heads dimensions can be thought of as a channel in the signal (like we have
            # RGB for images we have our heads-channels for the text).
            # We can perform a series of 1d convolutions on each channel to extract more
            # info from this representation, before passing it to the dense layers.
            # Note: we want to convolve the input until the third dimension becomes one
                (so that we obtain a flattening of the sequence)
            """
        # a few calculations to get the number of layers given the rest
        # since we're diving by 2 the dimension at each layer we need
        if not num_layers: num_layers = int(math.log2(input_length))
        convolutions = []
        channels = initial_filter_size
        padding = "same"
        for l in range(num_layers):
            if channels<threshold_channels//2 : channels = channels * 2
            layer = tf.keras.layers.Conv1D(filters=channels,
                                           kernel_size=window_size,
                                           strides=2,
                                           padding=padding,
                                           dilation_rate=dilation_rate,
                                           activation="relu",
                                           data_format="channels_first",  # data format : [batch, channels, signal]
                                           name='convolution{}'.format(str(l + 1)))
            convolutions.append(layer)
        # Flattening to 2D
        # Input shape:
        #     - If `data_format='channels_last'`:
        #       3D tensor with shape:
        #       `(batch_size, steps, features)`
        #   Output shape:
        #     2D tensor with shape `(batch_size, features)`.
        pooling = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")
        convolutions.append(pooling)
        return convolutions

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
        use_pretrained_embedding = kwargs.get("use_pretrained_embedding",True)
        dropout_rate = kwargs.get("dropout_rate",0.0) # setting the dropout to 0 is equivalent to not using it
        use_normalization = kwargs.get("use_normalization",False)
        activation = kwargs.get("activation","relu")
        metrics = kwargs.get("metrics",['accuracy'])
        optim = kwargs.get("optimizer","sgd")
        # CONVOLUTION parameters [only used is use_convolution is True]
        use_convolution = kwargs.get("use_convolution", True)
        num_conv_layers = kwargs.get("num_conv_layers")
        threshold_channels = kwargs.get("threshold_channels",600)
        dilation_rate = kwargs.get("dilation_rate", 1)  # note: dilation_rate = 1 means no dilation

        ## ---------------------
        # Note the shape parameter must not include the batch size
        # Here None stands for the timesteps
        inputs = tf.keras.layers.Input(shape=(50,), dtype='int32')
        weights = None
        if use_pretrained_embedding: weights = [self.embedding_matrix]
        embedding = tf.keras.layers.Embedding(input_dim=self.vocabulary_dim,
                                              output_dim=self.input_dim,
                                              weights=weights,
                                              mask_zero=True,
                                              trainable=train_embedding)
        masking = tf.keras.layers.Masking(mask_value=0)
        recurrent_cell = self.get_cell(cell_type)
        # recurrent layer
        # we return the entire sequence only if we want to convolve it
        recurrent_layer = tf.keras.layers.Bidirectional(recurrent_cell(hidden_size,
                                                                       return_sequences=use_convolution,
                                                                       recurrent_dropout=dropout_rate),
                                                        merge_mode="concat",
                                                        name="RecurrentLayer")
        if use_convolution:
            convolutions = self.build_convolutions(dilation_rate=dilation_rate,
                                                   initial_filter_size=hidden_size*2,
                                                   window_size=(hidden_size // 10),
                                                   input_length=50,
                                                   num_layers=num_conv_layers,
                                                   threshold_channels=threshold_channels)

        ## Dense head --------
        ## This last part of the model is responsible for mapping
        ## back the output of the recurrent layer to a binary value,
        ## which will indeed be our prediction
        dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        dense1 = tf.keras.layers.Dense(hidden_size, activation=activation, name="Dense1")
        if use_normalization: norm = tf.keras.layers.BatchNormalization()
        dense2 = tf.keras.layers.Dense(2, name="Dense2")
        ## CONNECTING THE GRAPH
        embedded_inputs = embedding(inputs)
        masked_inputs = masking(embedded_inputs)
        recurrent_output = recurrent_layer(masked_inputs)
        dropped_out = dropout(recurrent_output)
        flattened = tf.reshape(dropped_out, shape=[-1, hidden_size*2])
        if use_convolution:
            conv_result = dropped_out
            for conv_layer in convolutions:
                conv_result = conv_layer(conv_result)
            # the output of the last convolution layer will have size
            # [batch_size x num_channels]
            flattened = conv_result
        dense1_out = dense1(flattened)
        if use_normalization: dense1_out = norm(dense1_out)
        dense2_out = dense2(dense1_out)
        self.model = tf.keras.Model(inputs=inputs, outputs=dense2_out)
        optimizer = self.get_optimizer(optim)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics,
                           experimental_run_tf_function=False)
        self.model.summary()

class attention_NN(recurrent_NN):
    """Extends the recurrent model with self-attention.
    ------------------------
    More on self-attention:
    Our aim is to encode a variable length sentence into a fixed size embedding.
    We achieve that by choosing a linear combination of the [timesteps] LSTM hidden vectors
    in H. Computing the linear combination requires the self-attention mechanism.
    The attention mechanism takes the whole LSTM hidden states H as input,
    and outputs a vector of weights a. """
    def __init__(self,
                 embedding_dimension,
                 vocabulary_dimension,
                 embedding_matrix=None,
                 name="AttentionNN"):
        super().__init__(embedding_dimension=embedding_dimension,
                         vocabulary_dimension=vocabulary_dimension,
                         name=name,
                         embedding_matrix=embedding_matrix)


    def build(self, **kwargs):
        """Builds the attention NN computational graph"""
        print("Building model.")
        ## --------------------
        ## Extracting model parameters from arguments
        cell_type = kwargs.get("cell_type")  # either LSTM or GRU
        assert cell_type in self.possible_cell_values, "The cell type must be one of the following values : " \
                                                       + " ".join(self.possible_cell_values)  # printing the admissible cell values
        hidden_size = kwargs.get("hidden_size", 64)  # size of hidden representation
        train_embedding = kwargs.get("train_embedding", False)  # whether to train the Embedding layer to the model
        use_pretrained_embedding = kwargs.get("use_pretrained_embedding", True)
        dropout_rate = kwargs.get("dropout_rate", 0.0)  # setting the dropout to 0 is equivalent to not using it
        use_normalization = kwargs.get("use_normalization", False)
        activation = kwargs.get("activation", "relu")
        metrics = kwargs.get("metrics", ['accuracy'])
        optim = kwargs.get("optimizer", "sgd")
        d_a = kwargs.get("att_key_dim", 64)
        r = kwargs.get("heads",5) # the number of different parts to be extracted
                                        # from the sentence, i.e the number of aspects
                                        # to find in the sentence
        penalization = kwargs.get("penalization", True)
        gamma = kwargs.get("gamma",0.3) # penalization weight
        # CONVOLUTION parameters [only used is use_convolution is True]
        use_convolution = kwargs.get("use_convolution", False)
        dilation_rate = kwargs.get("dilation_rate", 1) #note: dilation_rate = 1 means no dilation

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
        query = tf.keras.layers.Dense(d_a, activation="tanh", name="attention1", use_bias=False)
        score = tf.keras.layers.Dense(r, name="attention2", use_bias=False)
        # CONVOLUTION LAYERS (to use instead of flattening)
        if use_convolution:
            convolutions = self.build_convolutions(dilation_rate=dilation_rate,
                                                                    initial_filter_size=r,
                                                                    window_size=(hidden_size//10),
                                                                    input_length=hidden_size*2)
        # DENSE head
        dense1 = tf.keras.layers.Dense(hidden_size, activation=activation, name="Dense1")
        if use_normalization: norm = tf.keras.layers.BatchNormalization()
        dense2 = tf.keras.layers.Dense(2, name="Dense2")
        ## CONNECTING THE GRAPH
        embedded_inputs = embedding(inputs)
        masked_inputs = masking(embedded_inputs)
        recurrent_output = recurrent_layer(masked_inputs) # tensor with all hidden states
                        # shape of recurrent_output = batch x timesteps x 2*hidden_size
        queries = query(recurrent_output) # W1 x H in the paper
        scores = score(queries) # batch x timesteps x r
        weights = tf.nn.softmax(scores, axis=1) # batch x timesteps x r
        # the weights we obtained sum to 1 for each of the r dimensions
        # we now use this weights to extract r different representations
        # of the sentence - hoping that they can capture different
        # aspects of the sentence
        # dimensions: [timesteps x r] x [timesteps x hidden_dim*2]
        #        --> [r x hidden_dim*2]
        context_vector = tf.matmul(weights,recurrent_output,
                                   transpose_a=True, name="apply_attention")
        dropped_out = dropout(context_vector)
        if use_normalization: dropped_out = norm(dropped_out)
        # flattening:
        # we now have our context vector with dimension [batch x r x hidden_dim*2]
        # we just flatten it (as done in the paper) to get a single vector
        # to work with
        flattened = tf.reshape(dropped_out, shape=[-1, r*hidden_size*2])
        ## Inserting convolution
        if use_convolution:
            conv_result = dropped_out
            for conv_layer in convolutions:
                conv_result = conv_layer(conv_result)
            # the output of the last convolution layer will have size
            # [batch_size x num_channels]
            flattened = conv_result
        dense1_out = dense1(flattened)
        outputs = dense2(dense1_out)
        ## COMPILING THE MODEL
        self.model = tf.keras.Model(inputs = inputs, outputs=outputs)
        optimizer = self.get_optimizer(optim)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # Here we insert the penalization term in the loss.
        # Any valid loss function needs to accept 2 terms.
        if penalization:
            def penalized_loss(y_true, y_pred):
                penalty = self.get_penalization(weights,gamma)
                cross_entr = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true,y_pred)
                return penalty + cross_entr
            loss = penalized_loss
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics,
                           experimental_run_tf_function=False)
        self.model.summary()

    @tf.function
    def get_penalization(self, weights, gamma):
        """ Defines the penalized version of the binary crossentropy loss.
                    --------------------------
                    More on penalization:
                    we are penalising on the distance from the identity matrix
                    of the dot product of our weight matrix by itself:
                    we are encouraging the dot product of two different
                    attention heads to be 0
                TODO: try with L1 norm instead of Frobenius
                """
        ## Compute penalization term
        res = tf.matmul(weights,weights, transpose_a=True) \
              - tf.eye(tf.shape(weights)[2])
        penalty = tf.norm(res, ord="fro", axis=[1, 2])**2  # frobenius norm of the residual matrix
        return gamma*tf.reduce_mean(penalty)

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

    def visualize_attention(self, sentence, sentence_vec):
        """ Helper function for interpretability of the model.
        :param sentence: (str) example sentence to use.
        :param sentence_vec: (list) list of indices representing the sentence.
        """
        from preprocessing.tokenizer import tokenize_text
        sentence = tokenize_text(sentence)
        weights = self.get_attention_weights(sentence_vec)
        prediction = self.model.predict(tf.expand_dims(sentence_vec, axis=0))
        probabilities = tf.nn.softmax(prediction, axis=1).numpy()[0]
        # Weights dimension: [1 x timesteps x r]
        # for each of the r dimensions we can plot the relevant words
        # we'll do this with a heat map over the sentence using
        # different colours
        weights = weights[:,0:len(sentence),:]
        weights = tf.reshape(weights, shape=weights.shape[1:]) # removing batch dimension
        weights = tf.transpose(weights) # shape = [r x words]
        # preparing the text to write on the heat_map
        labels = np.asarray(sentence*weights.shape[0])
        labels = labels.reshape(weights.shape)
        heat_map = sb.heatmap(weights, xticklabels=False, annot=labels, fmt='')
        plt.xlabel("Words")
        plt.ylabel("Attention heads")
        plt.title("Predictions : negative = {0}, "
                  "positive = {1}".format(probabilities[0],probabilities[1]), fontsize=10)
        plt.show()
