from transformers import BertTokenizer, TFBertModel
from base_NN import BaseNN
import tensorflow as tf


class HF_BERT_NN(BaseNN):
    """Neural net with Bert embedding
    layer from HuggingFace implementation."""

    def __init__(self,
                 embedding_dimension=-1,
                 name="HF_BERT_NN"):
        super().__init__(embedding_dimension, name=name)

    def bert_encoding(self, sentences):
        """Applies Bert encoding to a batch of sentences."""
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertModel.from_pretrained('bert-base-uncased')
        input_ids = tf.constant(tokenizer.encode(sentences)) # potentially not working
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        return last_hidden_states


    def build(self, **kwargs):
        optimizer = kwargs.get("optimizer", "adam")
        metrics = kwargs.get("metrics", ['accuracy'])
        dropout_rate = kwargs.get('dropout_rate', 0.5)

        ## BUILDING THE GRAPH
        inputs = tf.keras.layers.Input(shape=(None), name="inputs")
        # encode with Bert
        encoded_inputs = self.bert_encoding(inputs)
        dense_out_1 = tf.keras.layers.Dense(units=768, activation="tanh")(encoded_inputs)
        dense_out_1 = tf.keras.layers.Dropout(dropout_rate)(dense_out_1)
        dense_out_2 = tf.keras.layers.Dense(units=200, activation="softmax")(dense_out_1)
        dense_out_2 = tf.keras.layers.Dropout(dropout_rate)(dense_out_2)
        logits = tf.keras.layers.Dense(units=2, activation='softmax')(dense_out_2)

        self.model = tf.keras.Model(inputs=inputs, outputs=logits)
        self.model.build(input_shape=(None))
        self.model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=metrics)
        self.model.summary()
