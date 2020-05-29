# Classifiers 

Keeping track here of the experiments with different classifiers and their performances. 
<br>
<br>


--- 
### Vanilla NN 
Feed-forward neural network. <br>
Takes as input a sentence embedding. 
#### Experiment 1 
- **Structure**: 3 dense layers : <br>
    64 -> 64 -> 2 
- **Vocabulary used**: vocab.pkl
- **Embedding**: Glove
- **Using pre-trained embedding**: False
- **Performance**: <record here accuracy / f1 score> 
#### Experiment 2 
Same experiment as above, using pre-trained embedding. 
- **Using pre-trained embedding**: True 
- **Performance**: <record here accuracy / f1 score> 


--- 
### Recurrent net 
Takes as input the sentence as a sequence of tokens. Each token represents the index of 
the word in the vocabulary in use. The association of each word with its embedding is 
performed by an *Embedding layer* inside the network. The Embedding layer can be pre-loaded 
with a given embedding matrix. Furthermore, the embedding layer can be fine-tuned during 
training. 

#### Experiment 1 : "Recurrent_1L_GRU"
- **Structure**:

        |Embedding|| 
        ||Recurrence + 0.4 dropout|| 
        ||Dropout|| 
        ||Dense -64||
        ||Batch normalization||
        ||Dense -2||
        
        Note: using all that regularization because the net 
        was overfitting on the training set. 
   
- **Vocabulary used**: vocab.pkl
- **Embedding**: Glove
- **Using pre-trained embedding**: False
- **Trained embedding further**: False
- **Performance**: 
 
<div>
<img alt="accuracy" src="../data/assets/R_1L_GRU_acc.png" width="400"/>
<img alt="loss" src="../data/assets/R_1L_GRU_loss.png" width="400"/>
</div>

- **Training details**: 
        
        train_params={"epochs":15,
                        "batch_size":32,
                        "validation_split":0.3}
- **Other build details**: 

        build_params = {
                    "cell_type":"GRU",
                    "num_layers":1,
                    "hidden_size":64,
                    "optimizer":"adam",
                    "dropout_rate":0.4,
                    "use_normalization":True}

#### Experiment 2 : "Recurrent_1L_LSTM"
Same experiment as above, using LSTM cells instead of GRU.
- **Performance**: 

<div>
<img alt="accuracy" src="../data/assets/R_1L_LSTM_acc.png" width="400"/>
<img alt="loss" src="../data/assets/R_1L_LSTM_loss.png" width="400"/>
</div>

- **Other build details**: 

        build_params = {
                    "cell_type":"LSTM",
                    "num_layers":1,
                    "hidden_size":64,
                    "optimizer":"adam",
                    "dropout_rate":0.4,
                    "use_normalization":True}


#### Experiment 3 : "Attention_GRU"

The idea comes from [this](https://arxiv.org/pdf/1703.03130.pdf) paper. <br>
> Self-Attention (SA), a variant of the attention mechanism, 
>was proposed by Zhouhan Lin, et. al (2017) to overcome 
>the drawback of RNNs by allowing the attention mechanism to 
>focus on segments of the sentence, where the relevance of the 
>segment is determined by the contribution to the task.

- **Structure**:

<img src="https://miro.medium.com/max/1400/1*6c4-E0BRRLo197D_-vyXdg.png" width="500"/>

From part (a) in the above diagram, we can see the entire architecture of the self-attention model. <br>
The embedded tokens (w’s in the above) are fed into bidirectional LSTM layers (h’s). 
Hidden states are weighted by an attention vector (A’s) to obtain a refined sentence representation 
(M in the above) that is used as an input for the classification.
Part (b) of the diagram illustrates how to get the attention weights, proceeding from top to bottom. 
To begin with the collection of hidden states, it is multiplied by a weight matrix, and is f
ollowed by tanh layer for non-linear transformation. 
And then another linear transformation is applied to the output with another 
weight matrix to get the pre-attention matrix. A softmax layer, which is applied 
to the pre-attention matrix in the row-wise direction, making its weights looking 
like a probability distribution over the hidden states.
   
- **Vocabulary used**: vocab.pkl
- **Embedding**: Glove
- **Using pre-trained embedding**: False
- **Trained embedding further**: False
- **Performance**: 
 
<div>
<img alt="accuracy" src="../data/assets/ATT_GRU_acc.png" width="400"/>
<img alt="loss" src="" width="400"/>
</div>

- **Training details**:     

            train_params={"epochs":15,
                         "batch_size":32,
                         "validation_split":0.3}
                         
- **Other build details**: 

            build_params = {
                        "cell_type":"GRU",
                        "num_layers":1,
                        "hidden_size":64,
                        "optimizer":"adam",
                        "dropout_rate":0.4,
                        "use_normalization":True,
                        "use_attention":True, 
                        "heads":1}
         
                        
[Here](https://medium.com/apache-mxnet/sentiment-analysis-via-self-attention-with-mxnet-gluon-dc774d38ba69) is a link 
to a Medium article that explains the implementation of self-attention.


#### Experiment 4 : "Attention_GRU_5heads"

Same as the above experiment, using 5 attention heads instead of 1. 

        build_params = {
                "cell_type":"GRU",
                "num_layers":1,
                "hidden_size":64,
                "optimizer":"adam",
                "dropout_rate":0.4,
                "use_normalization":True,
                "use_attention":True,
                "heads":5, # number extracted from section 4.4.2 of the paper
                "penalization":False
            }
- **Performance**: 
 
<div>
<img alt="accuracy" src="../data/assets/ATT_GRU_5heads_acc.png" width="400"/>
<img alt="loss" src="../data/assets/ATT_GRU_5heads_loss.png" width="400"/>
</div>

- **Visualization**: 
Here's an heatmap visualization of the role of the 5 attention heads. <br>
Each row corresponds to a different attention head, and should hence capture a different "aspect" of the sentence.<br>
The heatmap shows the relative weights that each word is given by the attention head: the higher the weight, the more 
*"attention"* the word will receive at classification time.

<div>
<img alt="positive sentence heatmap" src="../data/assets/ATT_GRU_5heads_heatmap_pos.png" width="400"/>
<img alt="loss" src="../data/assets/ATT_GRU_5heads_heatmap_neg.png" width="400"/>
</div>

#### Experiment 4 : "Attention_GRU_5heads_penalized"

Same as the above, including penalization for the weight matrix A to encourage diversity. 
> The embedding matrix M can suffer from redundancy problems if the attention 
> mechanism always provides similar summation weights for all the r hops. 
> Thus we need a penalization term to encourage the diversity of summation 
> weight vectors across different hops of attention. 

            build_params = {
                    "cell_type":"GRU",
                    "num_layers":1,
                    "hidden_size":64,
                    "optimizer":"adam",
                    "dropout_rate":0.4,
                    "use_normalization":True,
                    "use_attention":True,
                    "heads":5, 
                    "penalization":True ## this is the difference
                }
                
            # More on penalization:
            # we are penalizing the distance from the identity matrix
            # of the dot product of our weight matrix by itself: 
            # we are encouraging the dot product of two different 
            # attention heads to be 0  
            --> disentangling the different factors of attention
   


--- 