#pipeline methods for preprocessing
from preprocessing.cooc import build_cooc, load_cooc
from preprocessing import cooc_folder

def get_lemmatization():
    pass

def get_vocabulary(vocabulary_name,
                   load_from_file=True,
                   input_files=None,
                   lemmatize_first=False):
    """
    Returns a word vocabulary.
    :param vocabulary_name: (str) the name of the vocabulary file.
            If `load_from_file` is False then this will be the name of the output
            file where the vocabulary will be stored.
            If `load_from_file` is True this should be the name of the file to load.
    :param load_from_file: (bool) Whether to load the vocabulary from file or build a new one.
    :param input_files: (list (str) ) Paths to the original files from which to extract
            the new vocabulary.
    :param lemmatize_first: (bool) whether to lemmatize the file before building the vocabulary
    :return: (dict) the vocabulary loaded/created
    """


    pass

def get_cooc_matrix(matrix_name,
                    vocab_name,
                    load_from_file=True,
                    input_files=None,
                    cooc_params=None):
    """
    Builds or loads a co-occurrence matriz from file. By default the matrix
    will be loaded from file.
    :param matrix_name: (str) the name of the cooc matrix file.
            If `load_from_file` is False then this will be the name of the output
            file where the cooc matrix will be stored.
            If `load_from_file` is True this should be the name of the file to load.
    :param vocab_name: (str) the name of the vocabulary to use to build the co-occurrence matrix
    :param load_from_file: (bool) Whether to load the cooc matrix from file or build a new one.
    :param input_files: (list (str) ) Paths to the original files from which to extract
            the new cooc matrix.
    :param cooc_params: (dict) dictionary of parameters to use in the `build_cooc` function
    :return: (np.ndarray)> the loaded co/occurrence matrix
    """

    # cooc_build parameters checking
    if cooc_params is None: cooc_params = {}
    window_size = cooc_params.get("window_size")
    weighting = cooc_params.get("weighting")
    if not weighting: weighting = "None"

    if load_from_file: cooc = load_cooc(cooc_folder+matrix_name)
    else: cooc = build_cooc(vocab_name=vocab_name,
                            window_size=window_size,
                            weighting=weighting,
                            output_name=matrix_name,
                            input_files=input_files)

    return cooc


def run_preprocessing():
    pass