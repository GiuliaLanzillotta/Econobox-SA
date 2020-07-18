from transformers import BertTokenizer
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import pandas as pd
import os
import numpy as np
from data import train_positive_location, train_negative_location
from data import replaced_train_full_positive_location, replaced_train_full_negative_location
from data import test_location
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def get_tweet_df(input_files, random_percentage):
    df_pos = pd.read_table(input_files[0], names=('tweet', 'id'))
    df_neg = pd.read_table(input_files[1], names=('tweet', 'id'))
    df_pos_1 = pd.DataFrame({
        'tweet': df_pos['tweet'],
        'label': 0
    })
    df_pos_1_s = df_pos_1.sample(n=int(df_pos_1.shape[0]*random_percentage))

    df_neg_1 = pd.DataFrame({
        'tweet': df_neg['tweet'],
        'label': 1
    })
    df_neg_1_s = df_neg_1.sample(n=int(df_neg_1.shape[0]*random_percentage))
    df_data = pd.concat([df_pos_1_s, df_neg_1_s], ignore_index=True, sort=False)
    df_data = shuffle(df_data)
    abs_path = os.path.abspath(os.path.dirname(__file__))
    #with open(os.path.join(abs_path,"../data/tweetDF.pkl"), 'wb') as f:
    #    pickle.dump(df_data,f, pickle.HIGHEST_PROTOCOL)

    return df_data

def get_tweet_df_pred(input_files):
    df_pred = pd.read_table(input_files,names=('tweet', 'id'))
    df_pred = pd.DataFrame({
        'tweet':df_pred['tweet']
    })
    abs_path = os.path.abspath(os.path.dirname(__file__))
    #with open(os.path.join(abs_path, "../data/predDF.pkl"), 'wb') as f:
    #    pickle.dump(df_pred, f, pickle.HIGHEST_PROTOCOL)
    return df_pred


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text



# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data, max_len):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=max_len,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,      # Return attention mask
            truncation=True
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks



def get_train_data(input_files, random_percentage, max_len):
    data = get_tweet_df(input_files=input_files, random_percentage=random_percentage)
    X = data.tweet.values
    y = data.label.values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1,
                                                      random_state=2020)
    train_inputs, train_masks = preprocessing_for_bert(data =X_train, max_len=max_len)
    val_inputs, val_masks = preprocessing_for_bert(data=X_val, max_len=max_len)

    #convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    #recommended batchsize
    batch_size = 32

    #DataLoader for training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    #DataLoader for validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader, y_train, y_val

def get_test_data(input_files, max_len):
    test_data = get_tweet_df_pred(input_files=input_files)

    test_inputs, test_masks = preprocessing_for_bert(data=test_data, max_len=max_len)

    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

    return test_dataloader



