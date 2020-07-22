import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from classifier.pipeline import run_train_pipeline, run_ensemble_pipeline, run_bert_torch_pipeline
#from data import train_positive_location, train_negative_location, test_location
from data import replaced_train_full_negative_location_30, replaced_train_full_positive_location_30, replaced_test_location
#from embedding import glove_30_matrix_train_location
from embedding import matrix_train_location
#from embedding import matrix_test_location
from data import replaced_train_full_positive_location_d, replaced_train_full_negative_location_d
from embedding import replaced_test_matrix_location

## ----------------
#Saving here the model specific parameters to pass to the
#model pipeline function.

recurrent_specific_params = {
    "vocabulary":"full_vocab_in_stanford.pkl",
    "load_embedding":True,
    "embedding_location":"necessary_stanford.npz",
    "generator_mode":True,
    "max_len":100
}


bert_specific_params = {
    "max_seq_len":128
}

rf_specific_params = {

}

HF_bert_specific_params = {

}

et_specific_params = {
    "number_of_embeddings":2,
    "vocabularies":["full_vocab_in_stanford.pkl","full_vocab_in_stanford.pkl"],
    "embedding_locations":["necessary_stanford.npz","glove_emb.npz"],
}
def bert_torch_main():
    run_bert_torch_pipeline(input_files_train=[replaced_train_full_negative_location_d, replaced_train_full_positive_location_d],
                            input_files_test=replaced_test_location,
                            name='BERT_torch_2_randsample',
                            random_percentage=0.3,
                            max_len=50,
                            epochs=6,
                            evaluation=True,
                            train_model=True,
                            load_model=False,
                            prediction_mode=True,
                            save_model=True
                            )

def ensemble_main():

   models = ["convolutional_NN","recurrent_NN"]
   models_names = ["convolution_3_pool","Attention_GRU_5heads_full"]
   data_locations = [replaced_zero_matrix_full_train_location,
                     replaced_zero_matrix_full_train_location]
   models_build_params = [
        # CONVOLUTIONAL PARAMETERS
       {
           "train_embedding": False,
           "use_pretrained_embedding": True,
           "use_pooling": True,
           "pooling_type": "max",
           "num_convolutions": 3,
           "window_size": 5,
           "dilation_rate": 1,  # no dilation
           "pool_size": 2,
           "hidden_size": 128,
           "dropout_rate": 0.4,
           "use_normalization": True,
           "optimizer": "adam"
       },
        # ATTENTION PARAMETERS
      {
           "cell_type":"GRU",
           "num_layers":1,
           "hidden_size":128,
           "optimizer":"adam",
           "dropout_rate":0.4,
           "use_normalization":True,
           "use_attention":True,
           "heads":5, # number extracted from section 4.4.2 of the paper
           "penalization":False
       }
   ]
   models_fun_params = [recurrent_specific_params,
                        recurrent_specific_params]
   run_ensemble_pipeline(models,
                         models_names,
                         data_locations,
                         models_build_params,
                         models_fun_params,
                         random_percentage=0.03,
                         prediction_mode=False,
                         ensemble_name="CONVATT_ensemble")


if __name__ == "__main__":
    #ensemble_main()
    bert_torch_main()


    """
    build_params = {"optimizer": 'adam',
                    "metrics": ['accuracy'],
                    "adapter_size": 1,
                    "train_embedding": False,
                    "use_pretrained_embedding": True,
                    "num_et_blocks": 1,
                    "max_len": 50}  # maximum length in the sequece

    train_params = {"epochs": 10,
                    "batch_size": 1024,
                    "validation_split": 0.2,
                    "use_categorical": True}

    run_train_pipeline("LR_classi",
                       "LR_baseline_sample_different_train",
                       load_model=True,
                       prediction_mode=True,
                       text_data_mode_on=False,
                       data_location=replaced_test_matrix_location,
                       cv_on=False,
                       choose_randomly=False,
                       random_percentage=1,
                       test_data_location=None,
                       generator_mode=False,
                       build_params=None,
                       train_params=None,
                       model_specific_params={})
                       
    """
                       

                       
                       

                       


    
