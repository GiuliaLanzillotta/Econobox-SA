from transformers import BertTokenizer, TFBertModel, AutoTokenizer
from base_NN import BaseNN
import tensorflow as tf
import numpy as np
#tf.compat.v1.disable_eager_execution()
from data import tweets_data
from data import train_negative_sample_location, train_positive_sample_location


class Lambda_Bert_Layer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Lambda_Bert_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.trans_model = TFBertModel.from_pretrained('bert-base-cased')


    def call(self, sentences, **kwargs):
        encoded_inputs = self.tokenizer.encode_plus(str(sentences), max_length=50, return_tensors='tf', padding=True,
                                                    truncation=True)
        output2 = self.trans_model(encoded_inputs['input_ids'], attention_mask=encoded_inputs['attention_mask'])
        return output2[:][-1][:]


    def compute_output_shape(self, input_shape):
        return (input_shape[0],1,768)

    def bert_autoencoding(self, sentences):
        """Applies Autotokenizer to a batch of sentences"""
        input_shape = tf.shape(sentences)
        # print(sentences)
        # print(tf.shape(sentences))
        # print(str(sentences))
        # print(type(str(sentences)))
        encoded_inputs = self.tokenizer.encode_plus(str(sentences), max_length=50, return_tensors='tf', padding=True,
                                                    truncation=True)
        # print("encoded_inputs", encoded_inputs)
        # print("input+ids", encoded_inputs['input_ids'])
        # print("attention_mask", encoded_inputs['attention_mask'])
        output2 = self.trans_model(encoded_inputs['input_ids'], attention_mask=encoded_inputs['attention_mask'])
        return output2[:][-1][:]


class HF_BERT_NN(BaseNN):
    """Neural net with Bert embedding
    layer from HuggingFace implementation."""

    def __init__(self,
                 max_len,
                 embedding_dimension=-1,
                 name="HF_BERT_NN"):
        super().__init__(embedding_dimension, name=name)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.trans_model = TFBertModel.from_pretrained('bert-base-cased')

    def bert_encoding(self, sentences):
        """Applies Bert encoding to a batch of sentences."""
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertModel.from_pretrained('bert-base-uncased')
        input_ids = tf.constant(tokenizer.encode((sentences)))  # potentially not working
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        return last_hidden_states

    @staticmethod
    def bert_autoencoding(sentences, max_len, tokenizer, trans_model):
        """Applies Autotokenizer to a batch of sentences"""
        # print(sentences)
        # print(tf.shape(sentences))
        # print(str(sentences))
        # print(type(str(sentences)))
        encoded_inputs = tokenizer.encode_plus(str(sentences), return_tensors='tf', max_length=max_len, truncation=True,
                                               padding=True)
        # print("encoded_inputs", encoded_inputs)
        # print("input+ids", encoded_inputs['input_ids'])
        # print("attention_mask", encoded_inputs['attention_mask'])
        output2 = trans_model(encoded_inputs['input_ids'], attention_mask=encoded_inputs['attention_mask'])
        return output2[:][-1][:]

    def build(self, **kwargs):
        optimizer = kwargs.get("optimizer", "adam")
        metrics = kwargs.get("metrics", ['accuracy'])
        dropout_rate = kwargs.get('dropout_rate', 0.5)

        ## BUILDING THE GRAPH
        input = tf.keras.layers.Input(shape=(None,), name='input', dtype=tf.string)  # This already gives shape None
        # lambda_layer = tf.keras.layers.Lambda(lambda x: self.bert_autoencoding(x, max_len=self.max_len, tokenizer=self.tokenizer, trans_model=self.trans_model), output_shape=(768), dynamic=True, trainable=False)(input)
        new_lambda_layer = Lambda_Bert_Layer(dynamic=True, trainable=False)
        lambda_output = new_lambda_layer(input)
        print("lambda output", lambda_output)
        # flat_lambda_layer = tf.keras.layers.Flatten(lambda_layer)
        reshape_lambda_layer = tf.reshape(lambda_output, (-1,768))
        dense_out_1 = tf.keras.layers.Dense(units=768, activation="tanh")(reshape_lambda_layer)  # reshape_lambda_layer
        dense_out_1 = tf.keras.layers.Dropout(dropout_rate)(dense_out_1)
        dense_out_2 = tf.keras.layers.Dense(units=200, activation="softmax")(dense_out_1)
        dense_out_2 = tf.keras.layers.Dropout(dropout_rate)(dense_out_2)
        logits = tf.keras.layers.Dense(units=2, activation='softmax')(dense_out_2)

        self.model = tf.keras.Model(inputs=input, outputs=logits)
        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=metrics)
        self.model.summary()

    def train(self, data_train, data_val, **kwargs):
        """
        Training the model and saving the history of training
        """
        epochs = kwargs.get("epochs", 10)
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                              min_delta=0.0001,
                                                              patience=1)

        self.history = self.model.fit(data_train,
                                      epochs=epochs,
                                      steps_per_epoch=int(2479999/2048),
                                      validation_data=data_val,
                                      callbacks=[earlystop_callback]
                                      )
        self.plot_history()

    def make_predictions(self, x, save=True, **kwargs):
        print("Making predictions")
        preds = self.model.predict(x, steps=10000)
        print(preds)
        preds_classes = np.argmax(preds, axis=-1).astype("int")
        preds_classes[preds_classes == 0] = -1
        if save: self.save_predictions(preds_classes)

#ourHFBert = HF_BERT_NN(max_len=50)
# output = ourHFBert.bert_autoencoding(sentences=["hello"])
# print("output1", output)
#ourHFBert.build()
#my_dataset = tweets_data.TweetDataset(input_files=[train_negative_sample_location, train_positive_sample_location],
#                                      labels=[0, 1], encode_text=False, do_padding=False, batch_size=1024)
# print(tf.shape(my_dataset.train))
#ourHFBert.train(data_train=my_dataset.train, data_val=my_dataset.validate)
#print(ourHFBert.model.summary())
