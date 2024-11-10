import os
import sys
sys.path.append("/Users/yamanmaharjan/Documents/Personal_yaman/Huggingface-summarizer_python/Modeling_task/src")
sys.path.append("/Users/yamanmaharjan/Documents/Personal_yaman/Huggingface-summarizer_python")

from sklearn.pipeline import Pipeline
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation







class The_Training_pipeline:


    def __init__(self):
        self.DataIngestion_object = DataIngestion()
        self.DataTrasformation_object = DataTransformation()
        
              
        self.pipeline = Pipeline(steps=[
            ('downloading_dataset', self.DataIngestion_object.initiate_data_Extraction), 
            ('transforming_dataset', self.DataTrasformation_object.Full_conversion_of_dataset) 
        ])

    def run_program(self):
           print('hello world')
           for step_name, step_function in self.pipeline.steps:
                  print('started', step_name)
                  step_function()
    

if __name__ == '__main__':
    main_pipeline = The_Training_pipeline()
    main_pipeline.run_program()


