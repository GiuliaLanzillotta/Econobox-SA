# Econobox-SA

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

#### Main.py
**Attention** As of 10.04.2020 running the main script will build the vocabulary and the co-occurrence matrix using the FULL files - I ran out of memory while doing that. 
