import os
import sys
sys.path.append("/Users/yamanmaharjan/Documents/Personal_yaman/NLP_Final_Yaman")
from dataclasses import dataclass
from NLP_MODULE.config import LoadConfig
import pandas as pd
from exception import CustomeException
from NLP_MODULE.config import LoadConfig
from logger import logging
import nltk
import ssl

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import string

@dataclass
class DataTransformationConfig:
    Load_data_main_path_root: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))))
    Loaded_data_path = os.path.join(Load_data_main_path_root,'Load_data_here','Loaded_dataset')
    Transformed_data_path = os.path.join(Load_data_main_path_root,'Load_data_here','Transformed_dataset')
    

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        self.stemmer =nltk.SnowballStemmer("english")
 
         
    def clean_imbalanced_data_(self):
         print("reached clean_imbalanced_data")
         try:
            config = LoadConfig.get_config_Full_file()
            imbalance_data = pd.read_csv(os.path.join(self.transformation_config.Loaded_data_path, 'imbalanced_data.csv'))
            print("The shape of imbalanced data is",imbalance_data.shape)
            imbalance_data.drop(config['ID'], axis=config['AXIS'], inplace=True)
            return imbalance_data
         except Exception as e:
            raise CustomeException(e, sys) from e
         

    def clean_raw_data_(self):
        print('Reached cleaning clean_raw_data')

        try:
            config = LoadConfig.get_config_Full_file()
            raw_data = pd.read_csv(os.path.join(self.transformation_config.Loaded_data_path, 'raw_data.csv'))
            raw_data.drop(["Unnamed: 0", 'count', 'hate_speech', 'offensive_language', 'neither'], axis=config['AXIS'], inplace=True)
            
            raw_data['class'] = raw_data['class'].replace({0: 1, 2: 0})
            raw_data.rename(columns={'class': 'label'}, inplace=True)
            return raw_data
        except Exception as e:
            raise CustomeException(e, sys) from e
    
    def concat_data_cleaning_of_words(self, words):
        try:  
            stemmer = nltk.SnowballStemmer("english")
            stopword = set(stopwords.words('english'))
            
            words = str(words).lower()
            words = re.sub(r'\n', ' ', words)  # Fixed regex for new lines
            words = re.sub(r'https?://\S+|www\.\S+', '', words)
            words = re.sub(r'<.*?>+', '', words)
            words = re.sub(r'[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub(r'\w*\d\w*', '', words)
    
    # Corrected stopword removal
            words = [word for word in words.split(' ') if word not in stopword]  
            words = " ".join(words)
            words = [stemmer.stem(word) for word in words.split(' ')]  # Changed to word instead of words
            words = " ".join(words)
            words = words.encode('ascii', 'ignore').decode('ascii')
            
    
            return words
               

        except Exception as e:
            raise CustomeException(e, sys) from e
    
    def generate_and_combine_imbalanced_and_raw_data_(self) -> None:
        print("generating combined training data from generate_and_combine_imbalanced_and_raw_data")
        try:
            cleaned_imbalanced_data = self.clean_imbalanced_data_()
            cleaned_raw_data = self.clean_raw_data_()

            combined_data = pd.concat([cleaned_imbalanced_data, cleaned_raw_data], ignore_index=True).astype(str)
            print("combined raw and imbalaced after cleaning")
            print("the columns are",combined_data.columns)
            if 'tweet' in combined_data.columns:
                
                
                combined_data['tweet'] = combined_data['tweet'].apply(self.concat_data_cleaning_of_words)
                combined_data['tweet'] = combined_data['tweet'].fillna(" ")
                combined_data['tweet'] = combined_data['tweet'].astype(str)
                print("The final transformed data is",combined_data.shape)
                combined_data.to_csv(os.path.join(self.transformation_config.Transformed_data_path, 'transformed_data.csv'), index=False)
            else:
                print('tweet not found')



        except Exception as e:
            raise CustomeException(e, sys) from e
        

    def apply_additional_cleaning_to_dataframe(self):
        pass