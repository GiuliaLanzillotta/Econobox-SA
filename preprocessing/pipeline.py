#pipeline methods for preprocessing
from preprocessing.cooc import build_cooc, load_cooc
from preprocessing.tokenizer import build_vocab, load_vocab
from preprocessing import cooc_folder
from preprocessing import lemmatizer as lemma
from data import input_files_location
import pickle
import os

def get_lemmatization(dictionary, stopwords = 0, output = "vocab_stemmed.pkl"):
    """
    Gives as output a dictionary that has been lemmatized, id believes -> belief and
    that has abbreviated words i.d. loove -> love.
    :param dictionary: (str) or (dict), either a dictionary or a file pkl containing a dictionary.
    :param stopwords: (bool) whether you want your dictionary to contain stopwords or not (0 to contain them) (1 to delete them)    
    :output: (str) ouput name
    
    """
    if(type(dictionary) != dict):
        with open(dictionary, "rb") as f:
            dictionary = pickle.load(f)
            
    voc_lemm =  lemma.DictionaryLemmatizer(dictionary = dictionary,stopword= stopwords,file_name = output)
    return voc_lemm
    
    
    

def get_vocabulary(vocabulary_name,
                   load_from_file=True,
                   input_files=None,
                   replacement_first=False,
                   vocab_params=None):
    """
    Returns a word vocabulary.
    :param vocabulary_name: (str) the name of the vocabulary file.
            If `load_from_file` is False then this will be the name of the output
            file where the vocabulary will be stored.
            If `load_from_file` is True this should be the name of the file to load.
    :param load_from_file: (bool) Whether to load the vocabulary from file or build a new one.
    :param input_files: (list (str) ) Paths to the original files from which to extract
            the new vocabulary.
    :param replacement_first: (bool) whether to obtain a vocabulary with words in short formed prolonged, such as
            "won't"->"will not", "don't"->"do not". For other lemmatizations, refer to the function get_lemmatization.
            lemmatize_first can be equal to True only if input_files is different from None.
    :param vocab_params: (dict) dictionary of parameters to use in the `build_vocab` function
            NOTA BENE: it is a good practice to leave all the specific params of a function
            in a general dict because it makes it easier in the future to modify them.
    :return: (dict) the vocabulary loaded/created
    """

    if vocab_params is None:
        vocab_params = {}
    frequency_treshold = vocab_params.get("frequency_treshold")

    if replacement_first:
        if(input_files is None):
            print("the argument input_files should not be None \n")
            print("Replacement will not be executed \n")
        else:
            print("function get_vocabulary: starting replacement (i.d don't -> do not ) \n")
            lemm_rep = lemma.RegexpReplacer()
            for i,input_file in enumerate(input_files):
                with open(input_file) as f:
                    file_lemmatized = lemm_rep.replace(f.read())
                with open("tobe_deleted_"+str(i + 1)+".txt", "w") as f:
                    f.write(file_lemmatized)
            input_files = ["tobe_deleted_1.txt","tobe_deleted_2.txt"]
            print("replacement done. \n")
            
            
        
    if load_from_file: vocab = load_vocab(vocabulary_name)
    else: vocab = build_vocab(frequency_treshold,
                              vocabulary_name,
                              input_files)
    if replacement_first:    
        os.remove("tobe_deleted_1.txt")
        os.remove("tobe_deleted_2.txt")
    return vocab



def getTxtLemmatization(input_files,
                        stopwords = False,
                        replace = True,
                        replace_stanford=False,
                        lemmatize = False,
                        outputfiles = None):
    """
    function that produces a lemmatized text.
    :param input_files: a list containing the path to the two files.
    :stopwords : (bool) either you want to have stopwords or not.
    :replacement : (bool) either you want to have replacement id. don't -> do not
    :outputfile: list containing the names of the two output files
    """
    abs_path = os.path.abspath(os.path.dirname(__file__))
    if not outputfiles: outputfiles = ["lemmatized_"+f for f in input_files]
    if replace:
        print("Starting replacement")
        mode = "standard"
        if replace_stanford: mode = "stanford"
        lemm_rep = lemma.RegexpReplacer(mode)
        for i,input_file in enumerate(input_files):
            with open(os.path.join(abs_path,input_files_location+input_file), encoding='utf8') as f:
                file_replaced = lemm_rep.replace(f.read())
            with open(os.path.join(abs_path,input_files_location+"replaced_"+input_file), "w",
                      encoding='utf8') as f:
                f.write(file_replaced)
        input_files = ["replaced_"+f for f in input_files]
        print("replacement done. \n")
    if lemmatize:
        for i,file_name in enumerate(input_files):
            print("Starting lemmatizing on"+file_name)
            lemma.TxtLemmatized(file_name=file_name,
                                  stopword=stopwords,
                                  output_file=outputfiles[0],
                                  use_lemmatizer=True,
                                  use_replacer=True)
    print("Done \n")
    if replace and lemmatize:
        os.remove(input_files[0], input_files[1])

        




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


def run_preprocessing(vocab_name,
                      cooc_name,
                      to_build_vocab=True,
                      to_build_cooc=True,
                      to_lemmatize_input=False, #if True we would lemmatize with stopwords = 0 and replacement = 1
                      vocab_build_params=None,
                      cooc_build_params=None,
                      input_files=None):
    """
    Runs the preprocessing steps using the pipeline functions
    :param vocab_name: (str) name of the vocabulary file (extension included)
    :param cooc_name: (str) name of the co-occurrence file ( // )
    :param to_build_vocab: (bool) whether to build the vocabulary from the @:param input_files
            or to load it from the @:param vocab_name file.
    :param to_build_cooc: (bool) whether to build the cooc matrix from the @:param input_files
            or to load it from the @:param cooc_name file.
    :param to_lemmatize_input: (bool) whether to lemmatize the input or not
    :param vocab_build_params: (dict) the parameters to be used by the  'build_vocab' function
    :param cooc_build_params: (dict) the parameters to be used by the   'build_cooc' function
    :param input_files:  (list(str)) the list of input files, if not provided the sample files will be used.
    :return: vocabulary and co-occurrence matrix
    """
    if(to_lemmatize_input):
        getTxtLemmatization(input_files, stopwords = False, replace=True, lemmatize=True,
                                    outputfile = ["lemm_pos.txt", "lemm_neg.txt"])
        input_files =  ["lemm_pos.txt", "lemm_neg.txt"]
    print("I am here \n")
    vocab = get_vocabulary(vocab_name,
                           load_from_file=not to_build_vocab,
                           input_files = input_files,
                           replacement_first=0, #it's already carried out in the function above
                           vocab_params=vocab_build_params)
    if(to_lemmatize_input):
        vocab = get_lemmatization(vocab, stopwords = 0, output = "dict_tmp.pkl") #get lemmatization continues with all other kinds of lemmatization
    
    cooc = get_cooc_matrix(cooc_name,
                           vocab_name,
                           load_from_file= not to_build_cooc,
                           input_files = input_files,
                           cooc_params=cooc_build_params)
    return vocab, cooc
