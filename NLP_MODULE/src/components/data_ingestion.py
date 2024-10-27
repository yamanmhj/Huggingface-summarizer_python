import os
import sys
sys.path.append("/Users/yamanmaharjan/Documents/Personal_yaman/NLP_Final_Yaman")

from dataclasses import dataclass
from Load_data_here.gcloud import GET_Data_From_GLOUD
from NLP_MODULE.config import LoadConfig
from NLP_MODULE.src.components.data_transformation import DataTransformation
from exception import CustomeException
import zipfile

@dataclass
class DataIngestionConfig:
    Load_data_main_path_root: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))))
    Loaded_data_path = os.path.join(Load_data_main_path_root,'Load_data_here','Loaded_dataset')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
"""
    def initiate_data_Extraction(self):
        print("Getting data from gcloud...")
        try:
           
            config = LoadConfig.get_config_Full_file()
            print(config)
            print(config['BUCKET_NAME'])
            print(config['ZIP_FILE_NAME'])
            print(self.ingestion_config.Loaded_data_path)
            GET_Data_From_GLOUD.sync_folder_from_gcloud(
                config['BUCKET_NAME'], 
                config['ZIP_FILE_NAME'], 
                self.ingestion_config.Loaded_data_path
            )
        except Exception as e:
            raise CustomeException(e,sys)
        
    def extract_zip_files(self):
            print('Extracting zip files')
            for file_name in os.listdir(self.ingestion_config.Loaded_data_path):
             if file_name.endswith('.zip'):
                  zip_file_path = os.path.join(self.ingestion_config.Loaded_data_path, file_name)
            # Extract the ZIP file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.ingestion_config.Loaded_data_path)
            print(f'Extracted: {file_name}')
"""





    