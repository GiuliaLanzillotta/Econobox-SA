""" Collection of utils for the dataset."""
import numpy as np
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import embedding


def random_split(input_path, split, shuffle=True):
    """
    Helper function to randomly split the data matrix into test and training.
        The matrices will be saved as:
        input_path_split1 , ... , input_path_splitn,
        where n = round(1/split_size)
    :param shuffle: boolean, whether to shuffle the data before splitting
    :param split: float btw 0 and 1 indicating the size of each split
    :param input_path: path to the input matrix (numpy compressed format)
    Note: not all the matrix has to be used.
    :return: None
    """
    abs_path = os.path.abspath(os.path.dirname(__file__))
    matrix = np.load(os.path.join(abs_path,input_path))['arr_0']
    # compute the size of the split
    m = matrix.shape[0]
    split_size = int(split * m)
    n_splits = int(1/split)
    print("Splitting the data in ",n_splits," splits, each with size ",split_size)
    # shuffling
    indices = range(m)
    if shuffle: indices = np.random.choice(m, m, replace=False)
    # creating the splts
    start = 0
    for s in range(n_splits+1):
        end = start+split_size
        if s==n_splits: split_indices = indices[start:] # no cutting for the last split
        else : split_indices = indices[start:end]
        split_data = matrix[split_indices, : ]
        # saving the data
        name=input_path[:-4]+"_split{0}".format(s)+".npz"
        np.savez(os.path.join(abs_path,name), split_data)
        # sliding the start pointer
        start += split_size

    print("Splitting done.")


if __name__ =="__main__":
    random_split(embedding.replaced_zero_matrix_full_train_location,split=0.2)



