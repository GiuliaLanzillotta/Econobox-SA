#Functions to lemmatize/remove stopwords from either the dictionary or the txt input file.
from preprocessing.tokenizer import tokenize_text,build_vocab
import pickle,os
import numpy as np
from preprocessing import train_positive_sample_location, train_negative_sample_location
from nltk.corpus import stopwords
import re
from nltk.corpus import wordnet
from nltk import  WordNetLemmatizer
from collections import Counter




class RepeatReplacer(object):
    """
    It is a class that allows to abbreviate words such as "loove" and "greaat" into their correspondent "love" and "great" while still retaining words like
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
    @param stopwords: either to to include stop words or not. 0 includes stopwords, 1 deletes them. The process of deleting stopwords takes time.
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
    
    
def  TxtLemmatized(file_name = "train_pos.txt", stopword = 0, output_file = "lemm_pos.txt"):
    """Function that lemmatizes the input file. To be used for example when creating the glove embedding with a dictionary that has been lemmatized.
       @param file_name: str 
       name of the file
       @param stopword: if 1 then it deletes also the stop words: very very slow
    
    """
    lemmatizer = WordNetLemmatizer()
    replacer = RepeatReplacer()
    with open(os.getcwd() + "\\data\\" + file_name,"r") as f: ### PROBABLY YOU HAVE TO CHANGE DIRECTORY HERE
        file_new = []
        for i,sentence in enumerate(f):
            if i % 10000 == 0:
                print(i)
            sentence_temporal = []
            for words in sentence:
                words = lemmatizer.lemmatize(words)
                words = replacer.replace(words)
                sentence_temporal.append(words)
            if(stopword == 1):
                print("Starting deleting stopwords from sentence " , i, "\n")
                sentence_temporal = [word for word in sentence_temporal if not word in stopwords.words()]
            string = ""
            string_1 = string.join(sentence_temporal)
            file_new.append(string_1)
    with open(output_file, "w") as f:
        f.writelines(file_new)
    return file_new
      
