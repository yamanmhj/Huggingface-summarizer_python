from components import model_trainer
import os
import sys
import keras
import pandas as pd
from dataclasses import dataclass
import exception as CustomeException
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
sys.path.append("/Users/yamanmaharjan/Documents/Personal_yaman/NLP_Final_Yaman")
from NLP_MODULE.src.components.model_trainer import ModelTrainer
from tensorflow import keras



from NLP_MODULE.config import LoadConfig

import pickle




@dataclass
class ModelEvaluationconfig:
     Load_data_main_path_root: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))))
     split_Training_dataset_path = os.path.join(Load_data_main_path_root,'Load_data_here','Training_dataset')
     model_path = os.path.join(Load_data_main_path_root,'artifacts','trained_models.h5')
     tokenizer_path = os.path.join(Load_data_main_path_root,'artifacts','tokenizer.pkl')
     

class ModelEvaluation:
     def __init__(self):
          self.modelEvaluationconfig = ModelEvaluationconfig()
          
          



     def Evaluate_model(self):
          try:
               config = LoadConfig.get_config_Full_file()
               with open(os.path.join(self.modelEvaluationconfig.split_Training_dataset_path, 'X_test.pkl'), 'rb') as file:
                     loaded_X_test_df = pickle.load(file)     

               with open(os.path.join(self.modelEvaluationconfig.split_Training_dataset_path, 'y_test.pkl'), 'rb') as file:
                     loaded_y_test_df = pickle.load(file)     

               
               print("-------------------------------- loaded X test before training", loaded_X_test_df.shape)
               print("-------------------------------- loaded Y test before ttraining", loaded_y_test_df.shape) 
    
               load_trained_model = keras.models.load_model(self.modelEvaluationconfig.model_path)
                     

               print("reached model evaluation")
               with open(self.modelEvaluationconfig.tokenizer_path, 'rb') as file:
                     Tokenizer_from_trained = pickle.load(file)
               
               
                   
               X_test_sequence = Tokenizer_from_trained.texts_to_sequences(loaded_X_test_df)
            
               sequences_matrix_X_test = pad_sequences(X_test_sequence, maxlen=config['MAX_LEN'])
              
              
               accuray = load_trained_model.evaluate(sequences_matrix_X_test, loaded_y_test_df)
               print('Test loss:', accuray)
               lstm_prediction = load_trained_model.predict(sequences_matrix_X_test)
               

            
          

          except Exception as e:
               raise CustomeException(e, sys) from e