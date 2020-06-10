import tensorflow_hub as hub
import tensorflow as tf
import bert
FullTokenizer = bert.bert_tokenization.FullTokenizer
from tensorflow.keras.models import Model       # Keras is the new high level API for TensorFlow
import math
import numpy as np
from data import sample_dimension, \
    train_negative_sample_location, train_positive_sample_location, test_location
from embedding import bert_matrix_train_location


def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


#Nota bene??:
#the embedding dimension of a tweet(sentence) is standard 768

def get_BERT_EMBEDDING(input_files, output_location):
    out_dim = 768
    #the total number of tweets
    N = 20000
    output = np.zeros((N, out_dim))


    for i, file in enumerate(input_files):

        with open(file) as f:
            for l, line in enumerate(f):
                max_seq_length = 128
                input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,name="input_word_ids")
                input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,name="input_mask")
                segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,name="segment_ids")
                bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=True)
                pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

                vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
                do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
                tokenizer = FullTokenizer(vocab_file, do_lower_case)

                stokens = tokenizer.tokenize(line)
                stokens = ["[CLS]"] + stokens + ["[SEP]"]
                #get the model inputs from the tokens
                input_ids = get_ids(stokens, tokenizer, max_seq_length)
                input_masks = get_masks(stokens, max_seq_length)
                input_segments = get_segments(stokens, max_seq_length)
                model = Model(inputs=[input_word_ids, input_mask, segment_ids],
                              outputs=[pooled_output, sequence_output])

                print(len(input_ids), len(input_masks), len(input_segments))

                pool_embs, all_embs = model.predict([[input_ids], [input_masks], [input_segments]])


                #now we need to store this into a matrix
                #look at how we did this with the embedding matrix

                #output dim is also 768 then

                output[i, :] = pool_embs
                if l % 10000 == 0:
                    print(l)
    np.savez(output_location, output)
    return output

get_BERT_EMBEDDING([train_positive_sample_location, train_positive_sample_location],bert_matrix_train_location )










