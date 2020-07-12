"""Here the tensorflow input pipeline is built.
ABOUT THE TweetDataset
---------------------------
The TweetDataset class implements the whole data pipeline (from text file to network input)
for a tensorflow model.
USAGE:
- To create a new dataset you need to specify the input files to read, and a vocabulary to use.
    The Dataset class will read the files line by line and create a dataset out of it.
    If you want to create a training or testing dataset (for which you have labels), then you
    should also pass a labels list as shown below.
        # build the dataset
        my_dataset = TweetDataset(input_files=[file1.txt, file2.txt],
                                    vocabulary="vocab.pkl",
                                    labels=[0,1], # 0 for the first file, 1 for the second
                                    # additional parameters
                                     batch_size=batch_size,
                                     buffer_size=buffer_size,
                                     max_len=100,
                                     validation_size=validation_size)
        # and finally train the net
        net.train(my_dataset, **train_params)
    Note: if you want to use the dataset for prediction you just need to set labels to None.
- To add a preprocessing step in the pipeline just add a function to this class that workds
    with tf.data.Dataset objects like the *encode_text* function.
    Note: by default the text is now encoded, i.e. each string token is transformed into a vocabulary index.
    To keep the data as text add a check to avoid using the encode_text function.
"""
import tensorflow as tf
import tensorflow_datasets as tfds
from preprocessing import standard_vocab_name
from preprocessing.pipeline import get_vocabulary
import data
import os


class TweetDataset():
    "Wrapper class for tf.data.Dataset object"

    def __init__(self, input_files,
                 labels=None,
                 batch_size=2048,
                 buffer_size=1000,
                 vocabulary=None,
                 max_len=100,
                 validation_size=0.2,
                 test_size=0.2,
                 encode_text=False,
                 do_padding=False,
                 caching=False,
                 size = data.full_dimension):
        """
        :param encode_text: (bool)
        :param max_len: maximum length of the input sequence
            :type max_len: int
        :param validation_size
            :type validation_size: float
        :param input_files: list of input files to use.
            :type input_files: list(str)
        :param batch_size
            :type batch_size: int
        :param buffer_size:
            :type buffer_size: int
            More on buffer_size.
            We designed the Dataset.shuffle() transformation to handle datasets
            that are too large to fit in memory. Instead of shuffling the entire dataset,
            it maintains a buffer of buffer_size elements, and randomly selects the next
            element from that buffer (replacing it with the next input element,
            if one is available).
        """
        self.input_files = input_files
        self.buffer_size = buffer_size
        self.labels = labels
        self.dataset = self.load_dataset(labels)
        self.do_padding = do_padding
        self.steps_per_epoch = None
        self.size = size
        if encode_text:
            if not vocabulary: vocabulary = standard_vocab_name
            self.vocab = get_vocabulary(vocabulary)
            self.encode_text(self.vocab)

        if self.labels:
            self.train, self.validate, self.test = self.split_dataset(batch_size, validation_size,
                                                                      test_size, max_len)
            if caching:
                print("Caching.")
                abs_path = os.path.abspath(os.path.dirname(__file__))
                paths = [os.path.join(abs_path, f) for f in
                         ["../data/train_data", "../data/valid_data", "../data/test_data"]]
                self.train = self.train.cache(paths[0])
                self.validate = self.validate.cache(paths[1])
                self.test = self.test.cache(paths[2])
        else:
            if self.do_padding:
                self.dataset = self.dataset.padded_batch(batch_size, padded_shapes=[max_len])
            self.pred_data = self.dataset.shuffle(self.buffer_size)

    def load_dataset(self, labels):
        """
        :param labels: list of labels.
            :type labels: list(int)
            If labels=None the dataset will be loaded in test mode.
            If labels!= None the length of labels has to match the length of the input_files list.
        :return:
        """
        print("Loading the dataset.")
        ## READING TEXT FILES
        abs_path = os.path.abspath(os.path.dirname(__file__))
        paths = [os.path.join(abs_path, f) for f in self.input_files]
        datasets = []
        for path in paths:
            datasets.append(tf.data.TextLineDataset(path))
        ## ADDING LABELS (if requested)
        if labels:
            assert len(labels) == len(self.input_files)
            for i in range(len(datasets)):
                datasets[i] = datasets[i].map(lambda line: (tf.expand_dims(line, axis=0),
                                                            tf.expand_dims(tf.one_hot(labels[i], 2), axis=0)))
        ##MERGING the datasets
        dataset_final = datasets[0]
        for i in range(len(datasets)-1):
            dataset_final = dataset_final.concatenate(datasets[i+1])

        if not labels: dataset_final = dataset_final.map(lambda line: (tf.expand_dims(line, axis=0)))
        ##SHUFFLE
        dataset_final = dataset_final.shuffle(self.buffer_size, reshuffle_each_iteration=False)
        return dataset_final

    def encode_text(self, vocabulary):
        """Helper function to transform string tokens into integers."""
        print("Encoding the dataset.")
        vocabulary_set = set(vocabulary.keys())
        encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

        # We want to use Dataset.map to apply this function to each element
        # of the dataset. Dataset.map runs in graph mode so you can't .map
        # this function directly: You need to wrap it in a tf.py_function.
        # The tf.py_function will pass regular tensors
        # (with a value and a .numpy() method to access it),
        # to the wrapped python function.

        # py_func doesn't set the shape of the returned tensors.
        # More on py function:
        # This function allows expressing computations in a TensorFlow graph
        # as Python functions. In particular, it wraps a Python function func
        # in a once-differentiable TensorFlow operation that executes
        # it with eager execution enabled.
        def encode_map_fn(text, label):
            encoded_text, label = tf.py_function(lambda t, l: (encoder.encode(t.numpy()), l),
                                                 inp=[text, label], Tout=(tf.int64, tf.float32))
            encoded_text.set_shape([None])
            label.set_shape([])
            return encoded_text, label

        def encode_map_fn_no_label(text):
            encoded_text, label = tf.py_function(lambda t: encoder.encode(t.numpy()),
                                                 inp=[text], Tout=tf.int64)
            encoded_text.set_shape([None])
            label.set_shape([])
            return encoded_text

        # finally applying the transformation word->idx on the dataset
        if self.labels:
            self.dataset = self.dataset.map(encode_map_fn)
        else:
            self.dataset = self.dataset.map(encode_map_fn_no_label)

    def split_dataset(self, batch_size, validation_size, test_size, max_len):
        """Splits the dataset in training and validation."""
        print("Splitting dataset.")
        # First calculating the splits
        validation_split = int(validation_size*self.size)
        test_split = int(test_size*self.size)
        train_split = self.size - validation_split - test_split
        print("Training on ",train_split,", validating on ",validation_split,", testing on ",test_split)
        # TRAIN
        self.steps_per_epoch = train_split//batch_size
        train_data = self.dataset.skip(validation_split + test_split).shuffle(self.buffer_size).take(train_split)
        if self.do_padding: train_data = train_data.padded_batch(batch_size, padded_shapes=([max_len], [1]))
        # VALID
        validation_data = self.dataset.take(validation_split)
        if self.do_padding: validation_data = validation_data.padded_batch(batch_size, padded_shapes=([max_len], [1]))
        # TEST
        test_data = self.dataset.skip(validation_split).take(test_split)
        if self.do_padding: test_data = test_data.padded_batch(batch_size, padded_shapes=([max_len], [1]))

        return train_data.prefetch(self.buffer_size), \
               validation_data.prefetch(self.buffer_size), \
               test_data.prefetch(self.buffer_size)
