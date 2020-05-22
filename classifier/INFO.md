# Classifiers 

Keeping track here of the experiments with different classifiers and their performances. 

### Vanilla NN 
Feed-forward neural network. <br>
Takes as input a sentence embedding. 
#### Experiment 1 
- Structure: 3 dense layers : <br>
    64 -> 64 -> 2 
- Vocabulary used: vocab.pkl
- Embedding: Glove
- Using pre-trained embedding: False
- Performance: <record here accuracy / f1 score> 
#### Experiment 2 
- Structure: 3 dense layers : <br>
    64 -> 64 -> 2 
- Vocabulary used: vocab.pkl
- Embedding: Glove
- Using pre-trained embedding: True 
- Performance: <record here accuracy / f1 score> 

### Recurrent net 
Takes as input the sentence as a sequence of tokens. Each token represents the index of 
the word in the vocabulary in use. The association of each word with its embedding is 
performed by an *Embedding layer* inside the network. The Embedding layer can be pre-loaded 
with a given embedding matrix. Furthermore, the embedding layer can be fine-tuned during 
training. 

#### Experiment 1 : "Recurrent_1L_GRU"
- Structure:

        |Embedding|| 
        ||Recurrence + 0.4 dropout|| 
        ||Dropout|| 
        ||Dense -64||
        ||Batch normalization||
        ||Dense -2||
        
        Note: using all that regularization because the net 
        was overfitting on the training set. 
   
- Vocabulary used: vocab.pkl
- Embedding: Glove
- Using pre-trained embedding: True 
- Trained embedding further: False
- Performance: 
![accuracy](../data/assets/R_1L_GRU_acc.png)
![loss](../data/assets/R_1L_GRU_loss.png)
- Training details: 
        
        train_params={"epochs":15,
                        "batch_size":32,
                        "validation_split":0.3}
- Other build details: 

        build_params = {
                    "cell_type":"GRU",
                    "num_layers":1,
                    "hidden_size":64,
                    "optimizer":"adam",
                    "dropout_rate":0.4,
                    "use_normalization":True}
                    
#### Experiment 2 : attention

- Structure:

        |Embedding|| 
        ||Recurrence + 0.4 dropout|| 
        ||Dropout|| 
        ||Dense -64||
        ||Batch normalization||
        ||Dense -2||
        
        Note: using all that regularization because the net 
        was overfitting on the training set. 
   
- Vocabulary used: vocab.pkl
- Embedding: Glove
- Using pre-trained embedding: True 
- Trained embedding further: False
- Performance: <record here accuracy / f1 score> 
![accuracy](accuracyhere)
![loss](losshere)
- Training details: 
- Other build details: 
- Attention mechanism: 
![attention](attentionpichere)
[Here](https://www.tensorflow.org/tutorials/text/nmt_with_attention) is a link to a Tensorflow 2 
tutorial on attention. 