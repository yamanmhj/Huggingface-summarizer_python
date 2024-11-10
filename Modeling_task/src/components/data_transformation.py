import os
from dataclasses import dataclass
from datasets import load_dataset
from transformers import pipeline, set_seed
import pandas as pd
from transformers import PegasusTokenizer



from transformers import AutoTokenizer, PegasusForConditionalGeneration
import logging

# Assuming this imports from your custom classes
from Modeling_task.config import LoadConfig
from exception import CustomeException

@dataclass
class DataTransformationConfig:
    Load_data_main_path_root: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))))
    Loaded_data_path = os.path.join(Load_data_main_path_root, 'Load_data_here', 'Loaded_dataset')
    Transformed_data_path = os.path.join(Load_data_main_path_root, 'Load_data_here', 'Transformed_dataset')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        self.config = LoadConfig.get_config_Full_file()
        
        # Load the tokenizer once during initialization
        self.tokenizer = PegasusTokenizer.from_pretrained(self.config['tokenizer_name'])
    def convert_data_to_features(self, example_batch):
        """
        Converts the input data batch into tokenized format using the tokenizer.
        """
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def Full_conversion_of_dataset(self):
        """
        Loads, transforms, and saves datasets.
        """
        file_names = ['train.json', 'test.json', 'val.json']

        for file_name in file_names:
            # Construct the full path to each dataset
            dataset_path = os.path.join(self.transformation_config.Loaded_data_path, file_name)

           
                # Load the dataset (use load_dataset)
            dataset = load_dataset('json', data_files=dataset_path)

                # Apply transformation (map the function to the dataset)
            dataset_transformed = dataset.map(self.convert_data_to_features, batched=True)

                # Ensure the transformed data directory exists
            save_path = os.path.join(self.transformation_config.Transformed_data_path, f"{file_name.split('.')[0]}_transformed")
             # Create directory if it doesn't exist

                # Save the transformed dataset
            dataset_transformed.save_to_disk(save_path)

          
            