import os
import sys
sys.path.append("/Users/yamanmaharjan/Documents/Personal_yaman/Huggingface-summarizer_python/Modeling_task/src")
sys.path.append("/Users/yamanmaharjan/Documents/Personal_yaman/Huggingface-summarizer_python")

from dataclasses import dataclass



import zipfile

@dataclass
class DataIngestionConfig:
    Load_data_main_path_root: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))))
    Loaded_data_path = os.path.join(Load_data_main_path_root,'Load_data_here', 'Loaded_dataset')

import os
import zipfile
import shutil

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_Extraction(self):
      
        print('Extracting zip files')
        print('root is', self.ingestion_config.Load_data_main_path_root)
        print('data path is', self.ingestion_config.Loaded_data_path)

        # Delete all files and folders except zip files
        for file_name in os.listdir(self.ingestion_config.Loaded_data_path):
            file_path = os.path.join(self.ingestion_config.Loaded_data_path, file_name)
            if not file_name.endswith('.zip'):
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Delete the file
                    print(f'Deleted file: {file_name}')
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete the directory
                    print(f'Deleted directory: {file_name}')

        # Now extract the ZIP files
        for file_name in os.listdir(self.ingestion_config.Loaded_data_path):
            if file_name.endswith('.zip'):
                zip_file_path = os.path.join(self.ingestion_config.Loaded_data_path, file_name)
                # Extract the ZIP file
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(self.ingestion_config.Loaded_data_path)
                print(f'Extracted: {file_name}')

            

            



