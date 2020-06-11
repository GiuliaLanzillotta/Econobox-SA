#Functions to lemmatize/remove stopwords from either the dictionary or the txt input file.
from preprocessing.tokenizer import tokenize_text,build_vocab
import pickle,os
import numpy as np
from data import train_positive_sample_location, train_negative_sample_location
from nltk.corpus import stopwords
import re
from nltk.corpus import wordnet
from nltk import  WordNetLemmatizer
from collections import Counter
from data import input_files_location





class RepeatReplacer(object):
    """
    It is a class that allows to abbreviate words such as "loove" and "greaat" into their correspondent 
    "love" and "great" while still retaining words like
    "goose" from being transformed into "gose".
    """
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'
    def replace(self, word):
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word


def DictionaryLemmatizer(dictionary,stopword = 0, file_name = "dictionary_stemmed.pkl"):
    """
    Takes as input a dictionary and stems it according to Word Net Lemmatizer
    @param dictionary: dict
        Input dictionary to have words as keys.
    @param file_name: str
        name of the file
    @param stopwords: either to to include stop words or not. 0 includes stopwords, 1 deletes them.
        The process of deleting stopwords takes time.
    @returns: list(str)
       New dictionary lemmatized.
    """
    dictionary_stemmed = []
    replacer = RepeatReplacer()
    lemmatizer = WordNetLemmatizer()
    for word in dictionary.keys():
        word = lemmatizer.lemmatize(word) 
        word = replacer.replace(word)        
        dictionary_stemmed.append(word)
    if(stopword == 1):
        print("Deleting stop words \n")
        dictionary_stemmed = [word for word in dictionary_stemmed if not word in stopwords.words()]
    voc = Counter(dictionary_stemmed)
    voc = dict(voc)
    final_dict = {k:i for i,k in enumerate(voc)}
    with open(file_name, "wb") as f:
        pickle.dump(final_dict, f, pickle.HIGHEST_PROTOCOL)
    return final_dict
    

replacement_patterns = [
(r'won\'t', 'will not'),
(r'can\'t', 'cannot'),
(r'i\'m', 'i am'),
(r'ain\'t', 'is not'),
(r'(\w+)\'ll', '\g<1> will'),
(r'(\w+)n\'t', '\g<1> not'),
(r'(\w+)\'ve', '\g<1> have'),
(r'(\w+)\'s', '\g<1> is'),
(r'(\w+)\'re', '\g<1> are'),
(r'(\w+)\'d', '\g<1> would')
]

# Replacement patterns that mimics the preprocessing on the Tweets emplyed by
# the Stanford group.
eyes = "[8:=;]"
nose = "['`\-]?"

def lower_replacement(match):
    return match.group(1).lower() + " <ALLCAPS>"

stanford_replacement_patterns = [
    (r'([A-Z]+)', lower_replacement),
    (r'https?://\S+\b|www\.\w+\.+\S*',"<URL>"), # matches an URL
    ("/"," / "),
    (r'@\w+', "<USER>"),
    (r''+eyes+''+nose+'?[)D]+|[(d]+'+nose+'?'+eyes, "<SMILE>"), # matces :-) or (-: and all variants
    (r''+eyes+nose+'?'+'p+|d+'+nose+'?'+eyes, "<LOLFACE>"), # matches :-p or d-: and all variants
    (r''+eyes+nose+'?'+'\(+|\)+'+nose+'?'+eyes, "<SADFACE>"),
    (r''+eyes+nose+'?'+'[\/|l*]|[\/|l*]+'+nose+'?'+eyes, "<NEUTRALFACE>"),
    (r'<3',"<HEART>"),
    (r'[-+]?[.\d]*[\d]+[:,.\d]*', "<NUMBER>"),
    (r'#([A-Z0-9]+)','<HASHTAG> \g<1> <ALLCAPS>'), # all caps hashtags
    (r'#(\w*[a-z]+\w*)','<HASHTAG> \g<1>'), # mixed hashtags
    (r'([!?.]){2,}','\g<1> <REPEAT>'), # mark punctuation repetition
    (r'\b(\S*?)(.)\2{2,}(.)*\b','\g<1>\g<2>\g<3> <ELONG>') # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
]

available_patterns = {
    "standard":replacement_patterns,
    "stanford":stanford_replacement_patterns
}

#Class that enables to change a string transforming grammatical forms like don't and can't in do not and cannot.
class RegexpReplacer(object):
    def __init__(self, patterns=None):
        self.patterns = available_patterns[patterns]
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s


def TxtLemmatized(file_name="train_pos.txt",
                  stopword=False,
                  output_file="lemm_pos.txt",
                  use_lemmatizer=True,
                  use_replacer=True):
    """Function that lemmatizes the input file. To be used for example when creating the glove
       embeddings with a dictionary that has been lemmatized.
       @param file_name: str
       name of the file
       @param stopword: if 1 then it deletes also the stop words: very very slow
       :param output_file: name of the output file
       :param patterns: either "standard" or "stanford".
                the patterns to be used by the regexpReplacer

    """
    if use_lemmatizer: lemmatizer = WordNetLemmatizer()
    if use_replacer: replacer = RepeatReplacer()
    abs_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(abs_path,input_files_location+file_name), encoding='utf8') as f:  ### PROBABLY YOU HAVE TO CHANGE DIRECTORY HERE
        file_new = []
        for i, sentence in enumerate(f):
            if i % 10000 == 0:
                print(i)
            sentence = tokenize_text(sentence)
            sentence_temporal = []
            for words in sentence:
                if use_lemmatizer: words = lemmatizer.lemmatize(words)
                if use_replacer: words = replacer.replace(words)
                sentence_temporal.append(words)
            if (stopword):
                print("Starting deleting stopwords from sentence ", i, "\n")
                sentence_temporal = [word for word in sentence_temporal if not word in stopwords.words()]
            string_1 = " ".join(sentence_temporal)
            file_new.append(string_1 + " \n")
    with open(os.path.join(abs_path,input_files_location+output_file), "w", encoding='utf8') as f:
        f.writelines(file_new)
    return file_new

