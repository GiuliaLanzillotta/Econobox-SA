# Embeddings

Just a recap on the Embeddings we have so far. 

1. *glove+stanford.npz* 
   
        get_glove_embedding(vocabulary_file="vocab.pkl",
                            load_from_file=False,
                            load_Stanford=True,
                            file_name="glove+stanford.npz",
                            train=True,
                            save=True)
                            
2. *glove_emb.npz*

        get_glove_embedding(vocabulary_file="vocab.pkl",
                            load_from_file=False,
                            load_Stanford=False,
                            file_name="glove_emb.npz",
                            train=True,
                            save=True)
                                
3. *only_stanford.npz*

           get_glove_embedding(vocabulary_file="stanford_vocab.pkl",
                                load_from_file=False,
                                load_Stanford=True,
                                file_name="only_stanford.npz",
                                train=False,
                                save=True)
                                
3. *necessary_stanford.npz*
    Stanford embedding restricted to the words in this dataset.
           get_glove_embedding(vocabulary_file="full_vocab.pkl",
                                load_from_file=False,
                                load_Stanford=True,
                                file_name="necessary_stanford.npz",
                                train=False,
                                save=True)