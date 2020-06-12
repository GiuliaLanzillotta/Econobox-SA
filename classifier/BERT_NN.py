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
from tqdm import tqdm
from data import tweetDF_location
from preprocessing.tweetDF import load_tweetDF
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from data import bert_ckpt_file_location
from data import bert_config_file_location
from data import bert_vocab_location
from preprocessing.tweetDF import load_tweetDF



class PP_BERT_Data():
    DATA_COLUMN = "text"
    LABEL_COLUMN = "sent"

    def __init__(self, train, classes, max_seq_length):
        self.max_seq_length = max_seq_length
        self.classes = classes
        self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True)
        self.tokenizer = FullTokenizer(vocab_file=bert_vocab_location)

        self.train_x, self.train_y = self._prepare(train)
        print(type(train))

        print("max seq_len", self.max_seq_length)
        #self.max_seq_length = min(self.max_seq_len, max_seq_lenght)
        self.train_x = self._pad(self.train_x)

    def _prepare(self, df):
        x, y = [], []

        for _, row in tqdm(df.iterrows()):
            text, label = row[PP_BERT_Data.DATA_COLUMN], row[PP_BERT_Data.LABEL_COLUMN]
            tokens = self.tokenizer.tokenize(text=text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.max_seq_length = max(self.max_seq_length, len(token_ids))
            x.append(token_ids)
            y.append(self.classes.index(label))

        return np.array(x), np.array(y)

    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_length - 2)]
            input_ids = input_ids + [0] * (self.max_seq_length - len(input_ids))
            x.append(np.array(input_ids))
        return np.array(x)


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
        self.classes = [0,1]


    def build(self, bert_ckpt_file, **kwargs):


        with tf.io.gfile.GFile(bert_config_file_location, "r") as reader:
            bc = StockBertConfig.from_json_string(reader.read())
            bert_params = map_stock_config_to_params(bc)
            bert_params.adapter_size = None
            bert = BertModelLayer.from_params(bert_params, name="bert")

        input_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype='int32', name="input_ids")
        bert_output = bert(input_ids)

        print("bert shape", bert_output.shape)

        cls_out = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
        cls_out = tf.keras.layers.Dropout(0.5)(cls_out)
        logits = tf.keras.layers.Dense(units=768, activation="tanh")(cls_out)
        logits = tf.keras.layers.Dropout(0.5)(logits)
        logits = tf.keras.layers.Dense(units=len(self.classes), activation="softmax")(logits)

        self.model = tf.keras.Model(inputs=input_ids, outputs=logits)
        self.model.build(input_shape=(None, self.max_seq_length))
        load_stock_weights(bert, bert_ckpt_file)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
        )


    def train(self, x, y,**kwargs):
        """
        Training the model and saving the history of training.
        """
        print("Training model.")
        epochs = kwargs.get("epochs")
        batch_size = kwargs.get("batch_size")
        validation_split = kwargs.get("validation_split")

        if not epochs: epochs = 1
        if not batch_size: batch_size=32
        if not validation_split: validation_split=0.20


        self.history = self.model.fit(
            x=x,
            y=y,
            validation_split=validation_split,
            batch_size=batch_size,
            shuffle=True,
            epochs=10,
        )
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

#BERT1 = BERT_NN(max_seq_length=128, embedding_dimension=200,name="BERT_NN")

#BERT1.build(bert_ckpt_file=bert_ckpt_file_location)

#data =
#dataset_train = data[:5]
#dataset_test = data[6:8]
#data = PP_BERT_Data(dataset_train, max_seq_length=128, classes=[0,1])
#x = data.train_x
#y = data.train_y

#BERT1.train(x,y)
#BERT1.make_predictions(x,save=False)


#BERT1.train()

#BERT1.train(dataset_train, dataset_test)
#ourdata = IntentDetectionData(dataset_train,classes=[0,1])
#print(type(ourdata))
#print(ourdata.train_x)


