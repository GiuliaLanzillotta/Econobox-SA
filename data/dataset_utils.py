""" Collection of utils for the dataset."""
import numpy as np
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import data


def random_split(input_path, split):
    """
    Helper function to randomly split a file into splits of size split*file_size.
        The files will be saved as:
        input_path_split1 , ... , input_path_splitn,
        where n = round(1/split_size)
    :param split: float btw 0 and 1 indicating the size of each split
    :param input_path: path to the input matrix (numpy compressed format)
    Note: not all the matrix has to be used.
    :return: None
    """
    abs_path = os.path.abspath(os.path.dirname(__file__))
    f = open(os.path.join(abs_path,input_path), encoding='utf8')
    lines = f.readlines()
    # compute the size of the split
    m = len(lines)
    split_size = int(split * m)
    n_splits = int(1/split)
    remainder = m%n_splits!=0
    if remainder: n_splits+=1
    print("Splitting the data in ",n_splits," splits, each with ",split_size, " lines.")
    # shuffling
    # creating the splts
    start = 0
    for s in range(n_splits):
        if s==n_splits and remainder: new_lines = lines[int(start):] # no cutting for the last split
        else : new_lines = lines[int(start):int(start + split_size)]
        # saving the new_lines
        name=input_path[:-4]+"_split{0}".format(s)+".txt"
        with open(os.path.join(abs_path,name), mode="w") as f_out:
            f_out.writelines(new_lines)
        # sliding the start pointer
        start += split_size

    print("Splitting done.")


if __name__ =="__main__":
    random_split(data.replaced_train_full_positive_location,split=0.2)



