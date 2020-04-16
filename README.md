<p align="center">
  <img src="https://github.com/GiuliaLanzillotta/Econobox-SA/blob/master/chevrolet_render.jpg" width="350" title="An econobox">
</p>


# Econobox 

### First things first.  
#### Dependencies
To use this repo you should install the dependencies that you find listed in the file *requirements.txt*. <br>
It'll be easy to do it if you use a virtual environment (recommended):
``` 
#install pipenv if you don't have it
pip install --user pipenv
pipenv install
```
To update the requirements file: 
``` 
pipenv lock -r > requirements.txt
```
However, if you don't want to set up a virtual environment you can still download the requirements as follows: 
```
# make sure to run this command in the project directory
pip install -r requirements.txt
```

#### Data 
To correctly run the code in this repo you need the data. <br>
There are two options here as well: <br>
1. Insert the **.zip** file containing the dataset inside the data directory and from your terminal run
    ```
    ./data/load_data.sh
    ```
2. Directly copy the unzipped folder containing the data inside the **/data** directory. 


## Now the structure. 
This repo is oganised in **4 modules** : 

    |_ Data 
    |   # here you can find all the code and stuff needed to 
    |   # load the data 
    |_ Preprocessing
    |   # Here you can find all the code responsible for the 
    |   # preprocessing of the input files. 
    |   # The preprocessing steps include: 
    |   #   - tokenize the words in the input
    |   #   - build a vocabulary
    |   #   - build a co-occurrence matrix
    |_ Embedding
    |   # Here you can find all the code responsible for the 
    |   # embedding of the tweets. 
    |   # The embedding can be done in two ways: 
    |   # - learn word embeddings -> find some fancy way to 
    |   #   aggregate the word embeddings into a sentence embedding
    |   # - learn the sentence embeddings directly
    |_ Classifier
    |   # Here you can find all the code responsible for the 
    |   # classification of the sentiment of tweets.
    |   # The classifier module is responsible for both training 
    |   # and testing. 
    
> #### The idea behind the modules structure 
> The goal of the re-factoring is to give each module a 
> simple (and hence easily extensible) internal structure. <br>

I am carrently still re-factoring the repo.
<br>Ideally each 
 module is defined by a **main**, an **init**, a **pipeline** and other
 **helper** scripts. The main should have little if no code, the init should
 define the constants that are needed across the module, the pipeline should
 define all the functions that access and organize the methods implemented
 in the helper scripts. The heavy logic of each implementation of the module
 should be in an helper script. 
 
 > #### How to extend the structure?
- Add a new *Pre-precessing method*: write 
it in a script and make sure to use it in the pipeline. 
- Add a new *Embedding method* : write a new subclass
of ```EmbeddingBase``` class, and implement its ```train_embedding``` 
method. 
- Add a new *Classifier method*: write a new subclass
of ```ClassifierBase``` class (IN-FIERI).

> #### What is still in-fieri? 
- [ ] Define pipeline for **Preprocessing**
- [x] Define pipeline for **Embedding**
- [x] Define pipeline for **Classifier**
- [x] Define **EmbeddingBase** class
- [x] Define **ClassifierBase** class
