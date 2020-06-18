"""
Classifier that uses evolved transformer blocks.

"""
from base_NN import BaseNN
import tensorflow as tf


class etransformer_NN(BaseNN):
    """
    Classifier based on evolved transformer blocks.

    """
    def __init__(self,
                 embedding_dimension,
                 vocabulary_dimension,
                 embedding_matrices=None,
                 number_of_embeddings=1,
                 name="etransformer_NN"):
        e_matrix=None
        if number_of_embeddings==1: e_matrix = embedding_matrices
        super().__init__(embedding_dimension=embedding_dimension,
                         vocabulary_dimension=vocabulary_dimension,
                         name=name,
                         embedding_matrix=e_matrix)
        #TODO: handle more than one embedding
        self.num_embeddings = number_of_embeddings

    @staticmethod
    def get_GLU(inputs):
        """
        Builds a Gated Linear Unit, according to
        https://michaelauli.github.io/papers/gcnn.pdf.
        :param filters: depth of the convolution.
        :param inputs: the input to the unit
        :return:
        """
        conv = tf.keras.layers.Conv1D(filters=256,
                                      kernel_size=3,
                                      padding="same",
                                      data_format="channels_last")
        X = conv(inputs)
        gates = tf.keras.activations.sigmoid
        out = X*gates(X)
        return out

    @staticmethod
    def get_Attention(inputs):
        """ Builds 8 heads attention block"""
        dense1 = tf.keras.layers.Dense(128, activation="tanh", use_bias=False)
        dense2 = tf.keras.layers.Dense(8, use_bias=False)
        dense1_res = dense1(inputs)
        scores = dense2(dense1_res)
        scores = tf.nn.softmax(scores, axis=1)
        context_vector = tf.matmul(scores, inputs,
                                   transpose_a=True)
        return context_vector

    def get_et_block(self, inputs):
        """Builds a new evolved transformer block."""
        norm = tf.keras.layers.LayerNormalization(axis=-1)
        ##----------
        # sub-block 1
        sub1_in = inputs
        normalized = norm(inputs)
        GLU = lambda x: self.get_GLU(x)
        GLU_res = GLU(normalized)
        sub1_out = sub1_in + GLU_res
        ## ---------
        # sub-block 2
        sub2_in = sub1_out
        normalized = norm(sub2_in)
        conv1 = tf.keras.layers.Conv1D(filters=2048,
                                       kernel_size=1,
                                       strides=1,
                                       padding="same",
                                       activation="relu")
        conv2 = tf.keras.layers.Conv1D(filters=256,
                                       kernel_size=3,
                                       strides=1,
                                       padding="same",
                                       activation="relu")
        conv1_out = conv1(normalized)
        conv2_out = conv2(normalized)
        # stacking the results of the two convolutions
        convs_out=tf.concat([conv1_out,conv2_out], axis=2)
        norm = tf.keras.layers.LayerNormalization(axis=-1)
        normalized = norm(convs_out)
        sep_conv = tf.keras.layers.SeparableConv1D(filters=256,
                                                   kernel_size=9,
                                                   strides=1,
                                                   padding="same",
                                                   data_format="channels_last")
        sep_conv_out = sep_conv(normalized)
        sub2_out = sub2_in + sep_conv_out
        ## -----------
        # sub-block 3
        sub3_in = sub2_out
        norm = tf.keras.layers.LayerNormalization(axis=-1)
        normalized = norm(sub3_in)
        attention = lambda x: self.get_Attention(x)
        attention_applied = attention(normalized)
        sub3_out = sub3_in + attention_applied
        ## -----------
        # sub-block 4
        sub4_in = sub3_out
        norm = tf.keras.layers.LayerNormalization(axis=-1)
        normalized = norm(sub4_in)
        deepen = tf.keras.layers.Conv1D(filters=2040,
                                        kernel_size=1,
                                        strides=1,
                                        padding="same",
                                        data_format="channels_last",
                                        activation="relu")
        deepened = deepen(normalized)
        shallow = tf.keras.layers.Conv1D(filters=512,
                                        kernel_size=1,
                                        strides=1,
                                        padding="same",
                                        data_format="channels_last")
        shallowed = shallow(deepened)
        dense256 = tf.keras.layers.Dense(256)
        sub4_out = sub4_in + dense256(shallowed)
        return sub4_out

    def build(self, **kwargs):
        print("Building model.")
        ## -------------------
        ## EXTRACTING ARGUMENTS
        train_embedding = kwargs.get("train_embedding", False)  # whether to train the Embedding layer to the model
        use_pretrained_embedding = kwargs.get("use_pretrained_embedding", True)
        num_et_blocks = kwargs.get("num_et_blocks",3)
        timesteps = kwargs.get("max_len",50)
        optim = kwargs.get("optimizer", "sgd")
        metrics = kwargs.get("metrics", ['accuracy'])
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
        dense1 = tf.keras.layers.Dense(256, activation="linear") # increase dimension to match et block dimension
        et_block = self.get_et_block
        dense2 = tf.keras.layers.Dense(256, activation="relu")
        dense_final = tf.keras.layers.Dense(2, activation="linear")
        ## -------------------
        ## CONNECTING THE NODES
        embedded_inputs = embedding(inputs)
        masked_inputs = masking(embedded_inputs)
        et_inputs = dense1(masked_inputs)
        for i in range(num_et_blocks):
            et_inputs = et_block(et_inputs)
        flattened = tf.reshape(et_inputs, shape=[-1,256*timesteps])
        dense2_out = dense2(flattened)
        outputs = dense_final(dense2_out)
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