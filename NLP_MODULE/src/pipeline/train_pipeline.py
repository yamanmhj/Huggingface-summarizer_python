import os
import sys
sys.path.append("/Users/yamanmaharjan/Documents/Personal_yaman/NLP_Final_Yaman/NLP_MODULE/src")
sys.path.append("/Users/yamanmaharjan/Documents/Personal_yaman/NLP_Final_Yaman")

from sklearn.pipeline import Pipeline
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_evaluation import ModelEvaluation
from components.model_trainer import ModelTrainer


class The_Training_pipeline:


    def __init__(self):
        
        self.pipeline = Pipeline(steps=[
            ('downloading_dataset', DataIngestion().initiate_data_Extraction),
            ('extracting zipline', DataIngestion().extract_zip_files),
            ('word processing and combining data',DataTransformation().generate_and_combine_imbalanced_and_raw_data_),
            ('Trainign model(splitting,tokenizer,modelname,)', ModelTrainer().initiate_model_training),
            ('Final model performance evaluation', ModelEvaluation().Evaluate_model)
        ])

    def run_program(self):
            for step_name, step_function in self.pipeline.steps:
                  #logging.info(f'Running step: {step_name}')
                  step_function()

    

if __name__ == '__main__':
    main_pipeline = The_Training_pipeline()
    main_pipeline.run_program()

