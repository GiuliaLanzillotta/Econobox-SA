from typing import Any, Union

from transformers import BertTokenizer, TFBertModel, AutoTokenizer, BertJapaneseTokenizer, \
    TFBertForSequenceClassification
from base_NN import BaseNN
import tensorflow as tf
import numpy as np

class Lambda_Bert_Layer(tf.keras.layers.Layer):


    def __init__(self, **kwargs):
        super(Lambda_Bert_Layer, self).__init__(**kwargs)
        self.tokenizer = None
        self.trans_model = None

    def build(self, input_shape):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.trans_model = TFBertModel.from_pretrained('bert-base-cased')

    def get_embedding(self, sentences):
        """Doing the real Bert job"""

        def convert_to_str(sentences):
            return [list(b) for b in sentences.numpy().astype(str)]

        encoded_input = self.tokenizer.batch_encode_plus(convert_to_str(sentences),
                                                         max_length=50,
                                                         return_tensors='tf',
                                                         add_special_tokens=True,
                                                         return_token_type_ids=False,
                                                         pad_to_max_length=True,
                                                         return_attention_mask=True)
        output = self.trans_model(encoded_input["input_ids"],
                                  attention_mask=encoded_input["attention_mask"])
        last_layer_states = output[0]
        last_layer_states = last_layer_states[:, -1, :]
        return last_layer_states # shape = (1*768)

    def call(self, sentences, **kwargs):
        outputs = self.get_embedding(sentences)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0],768)


class HF_BERT_NN(BaseNN):
    """Neural net with Bert embedding
    layer from HuggingFace implementation."""

    def __init__(self,
                 max_len,
                 embedding_dimension=-1,
                 name="HF_BERT_NN"):
        super().__init__(embedding_dimension, name=name)
        self.max_len = max_len



    def build(self, **kwargs):
        optimizer = kwargs.get("optimizer", "adam")
        metrics = kwargs.get("metrics", ['accuracy'])
        dropout_rate = kwargs.get('dropout_rate', 0.5)

        ## BUILDING THE GRAPH
        input_ids = tf.keras.layers.Input(shape=(1,50), name='input_ids', dtype=tf.int32)
        input_mask = tf.keras.layers.Input(shape=(1,50), name='input_mask', dtype=tf.int32)
        #bert_layer = Lambda_Bert_Layer(trainable=False, dynamic=True)
        bert_layer = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
        bert_layer.bert.trainable=False
        bert_output = bert_layer(input_ids[:,0,:], attention_mask=input_mask[:,0,:])
        """
        bert_layer.trainable = False
        bert_output=bert_output[0][:,-1,:]
        last_state = tf.reshape(bert_output, shape=(-1,768))
        dense_out_1 = tf.keras.layers.Dense(units=768, activation="relu")(last_state)  # reshape_lambda_layer
        dense_out_1 = tf.keras.layers.Dropout(dropout_rate)(dense_out_1)
        dense_out_2 = tf.keras.layers.Dense(units=200, activation="relu")(dense_out_1)
        dense_out_2 = tf.keras.layers.Dropout(dropout_rate)(dense_out_2)
        logits = tf.keras.layers.Dense(units=2, activation='softmax')(dense_out_2)
        """
        logits = bert_output[0]

        self.model = tf.keras.Model(inputs=(input_ids,input_mask), outputs=logits)
        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=metrics,
                           run_eagerly=True)
        self.model.summary()

    def train(self, data_train, data_val, **kwargs):
        """
        Training the model and saving the history of training
        """
        steps_per_epoch = kwargs.get("steps_per_epoch")
        valid_steps = kwargs.get("valid_steps")
        epochs = kwargs.get("epochs", 10)
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                              min_delta=0.0001,
                                                              patience=1)
        self.history = self.model.fit(data_train,
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      validation_data=data_val,
                                      validation_steps = valid_steps,
                                      #callbacks=[earlystop_callback]
                                      )
        self.plot_history()

    def make_predictions(self, x, save=True, **kwargs):
        print("Making predictions")
        preds = self.model.predict(x, steps=10000)
        preds_classes = np.argmax(preds, axis=-1).astype("int")
        preds_classes[preds_classes == 0] = -1
        if save: self.save_predictions(preds_classes)