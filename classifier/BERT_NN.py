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
from classifier.base_NN import BaseNN
from tqdm import tqdm
from data import tweetDF_location
from preprocessing.tweetDF import load_tweetDF
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from data import bert_ckpt_file_location
from data import bert_config_file_location
from data import bert_vocab_location
from preprocessing.tweetDF import load_tweetDF
abs_path = os.path.abspath(os.path.dirname(__file__))
import math



#code inspired on https://medium.com/analytics-vidhya/bert-in-keras-tensorflow-2-0-using-tfhub-huggingface-81c08c5f81d8
def flatten_layers(root_layer):
    if isinstance(root_layer, tf.keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer


def freeze_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False

#code inspired on https://medium.com/analytics-vidhya/bert-in-keras-tensorflow-2-0-using-tfhub-huggingface-81c08c5f81d8
class PP_BERT_Data():
    DATA_COLUMN = "text"
    LABEL_COLUMN = "sent"

    def __init__(self, data_input, classes, max_seq_length, prediction_mode):
        self.max_seq_length = max_seq_length
        self.classes = classes
        self.prediction_mode = prediction_mode
        print("self prediction mode", self.prediction_mode)
        self.tokenizer = FullTokenizer(vocab_file=os.path.join(abs_path,bert_vocab_location))

        if self.prediction_mode:
            self.pred_x = self._prepare(data_input)
            print("max seq_length", self.max_seq_length)
            self.pred_x = self._pad(self.pred_x)

        if not self.prediction_mode:
            self.train_x, self.train_y = self._prepare(data_input)
            print("max seq_length", self.max_seq_length)
            self.train_x = self._pad(self.train_x)


    def _prepare(self, df):
        print("Preparing dataframe...")
        x, y = [], []

        if not self.prediction_mode:
            for _, row in tqdm(df.iterrows()):
                text, label = row[PP_BERT_Data.DATA_COLUMN], row[PP_BERT_Data.LABEL_COLUMN]
                tokens = self.tokenizer.tokenize(text=text)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                x.append(token_ids)
                y.append(self.classes.index(label))

            return np.array(x), np.array(y)

        if self.prediction_mode:
            for _, row in tqdm(df.iterrows()):
                text = row[PP_BERT_Data.DATA_COLUMN]
                tokens = self.tokenizer.tokenize(text=text)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                x.append(token_ids)

            return np.array(x)


    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_length - 2)]
            input_ids = input_ids + [0] * (self.max_seq_length - len(input_ids))
            x.append(np.array(input_ids))
        return np.array(x)


class BERT_NN(BaseNN):
    """Neural net with Bert embedding layer"""

    def __init__(self,
                 max_seq_length,
                 adapter_size,
                 embedding_dimension,
                 name="BERT_NN"):
        super().__init__(embedding_dimension, name=name)
        self.history = None
        self.model = None
        self.max_seq_length = max_seq_length
        self.classes = [0,1]
        self.adapter_size = adapter_size

    def build(self, max_seq_length, bert_ckpt_file=bert_ckpt_file_location, **kwargs):
        optimizer = kwargs.get("optimizer", "adam")
        metrics = kwargs.get("metrics", ['accuracy'])
        adapter_size = kwargs.get("adapter_size", 64)
        dropout_rate = kwargs.get('dropout_rate', 0.5)


        # adapter_size = 64  # see - arXiv:1902.00751

        # create the bert layer
        with tf.io.gfile.GFile(os.path.join(abs_path,bert_config_file_location), "r") as reader:
            bc = StockBertConfig.from_json_string(reader.read())
            bert_params = map_stock_config_to_params(bc)
            bert_params.adapter_size = adapter_size
            bert = BertModelLayer.from_params(bert_params, name="bert")

        input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name="input_ids")
        output = bert(input_ids)

        print("bert shape", output.shape)
        cls_out = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
        cls_out = tf.keras.layers.Dropout(0.5)(cls_out)
        dense_out_1 = tf.keras.layers.Dense(units=768, activation="tanh")(cls_out)
        dense_out_1 = tf.keras.layers.Dropout(dropout_rate)(dense_out_1)
        dense_out_2 = tf.keras.layers.Dense(units=200, activation="softmax")(dense_out_1)
        dense_out_2 = tf.keras.layers.Dropout(dropout_rate)(dense_out_2)
        logits = tf.keras.layers.Dense(units=len(self.classes), activation='softmax')(dense_out_2)

        self.model = tf.keras.Model(inputs=input_ids, outputs=logits)
        self.model.build(input_shape=(None, max_seq_length))

        # load the pre-trained model weights
        load_stock_weights(bert, os.path.join(abs_path, bert_ckpt_file))

        # freeze weights if adapter-BERT is used
        if adapter_size is not None:
            freeze_bert_layers(bert)

        self.model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=metrics)

        self.model.summary()




