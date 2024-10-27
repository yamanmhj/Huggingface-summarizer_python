import os
from dataclasses import dataclass
from exception import CustomeException
import sys
import pandas as pd
import pickle
from logger import logging
from NLP_MODULE.config import LoadConfig
from NLP_MODULE.src.components.SELECT_MODEL import SelectModelArchitecture
from keras_preprocessing.text import Tokenizer

from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


sys.path.append("/Users/yamanmaharjan/Documents/Personal_yaman/NLP_Final_Yaman")


@dataclass
class ModelTrainerConfig: 
   Load_data_main_path_root: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))))
   Transformed_data_path = os.path.join(Load_data_main_path_root,'Load_data_here','Transformed_dataset')
   split_Training_dataset_path = os.path.join(Load_data_main_path_root,'Load_data_here','Training_dataset')
   model_path = os.path.join(Load_data_main_path_root,'artifacts','trained_models.h5')
   tokenizer_path = os.path.join(Load_data_main_path_root,'artifacts','tokenizer.pkl')

class ModelTrainer:
   
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()
        self.temp = []

    def Split_train_test_dataset(self):
        try:
            Final_cleaned_data_path = os.path.join(self.trainer_config.Transformed_data_path,'transformed_data.csv')
            training_dataset_df = pd.read_csv(Final_cleaned_data_path)
            training_dataset_df.dropna(inplace=True)
            training_dataset_df.fillna("Nones" ,inplace=True)
            print("shape after dropping",training_dataset_df.shape)
            
            
            config = LoadConfig.get_config_Full_file()

            x = training_dataset_df[config['TWEET']]
            y = training_dataset_df[config['LABEL']]

            X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.3 ,random_state = 42)

            print("--------------------------X_train_shape",X_train.shape)
            print("--------------------------X_test_shape",X_test.shape)
            print("--------------------------X_train_shape",y_train.shape)
            print("--------------------------y_shape",y_test.shape)

            with open(os.path.join(self.trainer_config.split_Training_dataset_path, 'X_train.pkl'), 'wb') as file:
                pickle.dump(X_train, file)

            with open(os.path.join(self.trainer_config.split_Training_dataset_path, 'X_test.pkl'), 'wb') as file:
                pickle.dump(X_test, file)

            with open(os.path.join(self.trainer_config.split_Training_dataset_path, 'y_train.pkl'), 'wb') as file:
                pickle.dump(y_train, file)
        
            with open(os.path.join(self.trainer_config.split_Training_dataset_path, 'y_test.pkl'), 'wb') as file:
                pickle.dump(y_test, file)



            
            return X_train,y_train

        except Exception as e:
            raise CustomeException(e, sys) from e
        

    def Tokenizing_the_dataset(self,data):
        try:
            config = LoadConfig.get_config_Full_file()
            print("Tokenizing the training dataset")
            tokenizer = Tokenizer(num_words = config['MAX_WORDS'])
            tokenizer.fit_on_texts(data)
            with open(self.trainer_config.tokenizer_path, 'wb') as file:
                pickle.dump(tokenizer, file)

            
            print("Generating sequence")
            sequences = tokenizer.texts_to_sequences(data)
            
            sequences_matrix = pad_sequences(sequences,maxlen=config['MAX_LEN'])
            return sequences_matrix
        except Exception as e:
            raise CustomeException(e, sys) from e

    

    def initiate_model_training(self) -> None:
        try:
            config = LoadConfig.get_config_Full_file()
            print("Starting model training")
            X_train,y_train = self.Split_train_test_dataset()
            model_selector = SelectModelArchitecture()
            NLP_model = model_selector.get_model_architecture()
            sequences_matrix = self.Tokenizing_the_dataset(X_train)
            
            print("Training the model")
            trained_model = NLP_model.fit(sequences_matrix, y_train, epochs=config['EPOCHS'], batch_size=config['BATCH_SIZE'], validation_split=config['VALIDATION_SPLIT'])   
            print("Training model completed")

            print('saving trained model')    
            NLP_model.save(self.trainer_config.model_path) 
            
            
        except Exception as e:
            raise CustomeException(e, sys) from e

    