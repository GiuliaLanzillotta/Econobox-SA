#pipeline methods for preprocessing

def get_vocabulary(vocabulary_name,
                   load_from_file=True,
                   original_texts=None,
                   lemmatize_first=False):
    """
    Returns a word vocabulary.
    :param vocabulary_name: (str) the name of the vocabulary file.
            If `load_from_file` is False then this will be the name of the output
            file where the vocabulary will be stored.
            If `load_from_file` is True this should be the name of the file to load.
    :param load_from_file: (bool) Whether to load the vocabulary from file or build a new one.
    :param original_texts: (list (str) ) Paths to the original files from which to extract
            the new vocabulary.
    :param lemmatize_first: (bool) whether to lemmatize the file before building the vocabulary
    :return: (dict) the vocabulary loaded/created
    """
    pass

def get_lemmatization():
    pass

def get_cooc_matrix():
    pass

def run_preprocessing():
    pass