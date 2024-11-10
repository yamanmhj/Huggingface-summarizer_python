import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
from datasets import load_from_disk, load_dataset
import sys

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" 

# Add custom module path for loading configurations
sys.path.append("/Users/yamanmaharjan/Documents/Personal_yaman/Roberta")
from NLP_MODULE.config import LoadConfig

# Paths for pre-transformed data
class ModelTrainerConfig:
    BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))))
    
    DATA_PATH = os.path.join(BASE_PATH, 'Load_data_here', 'Transformed_dataset')
    loaded_train_data_path = os.path.join(DATA_PATH, 'train_transformed')
    loaded_test_data_path = os.path.join(DATA_PATH, 'test_transformed')
    loaded_val_data_path = os.path.join(DATA_PATH, 'val_transformed')
    save_trained_model = os.path.join(BASE_PATH, 'artifacts','final_trained_model')
    save_tokenizer_full_token = os.path.join(BASE_PATH, 'artifacts','final_tokenizer_model')
    output_directory = os.path.join(BASE_PATH, 'artifacts','output_directory')

# Main Trainer class
class ModelTrainer:
    def __init__(self):
        self.model_training_config = ModelTrainerConfig()
        self.config = LoadConfig.get_config_Full_file()
       

    def initiate_train_model(self):
        # Device selection
        print("----------",self.model_training_config.BASE_PATH)
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer_name'], model_max_length=512)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config['tokenizer_name']).to(device)
        model.gradient_checkpointing_enable()
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        train_dataset = load_from_disk(ModelTrainerConfig.loaded_train_data_path)
        test_dataset = load_from_disk(ModelTrainerConfig.loaded_test_data_path)

        
        val_dataset = load_from_disk(ModelTrainerConfig.loaded_val_data_path)
        print("-----------1",ModelTrainerConfig.loaded_train_data_path)
        print("-----------1",ModelTrainerConfig.loaded_test_data_path)
        print("-----------1",ModelTrainerConfig.loaded_val_data_path)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir = self.model_training_config.output_directory,
            num_train_epochs=self.config['num_train_epochs'],
            warmup_steps=self.config['warmup_steps'],
            per_device_train_batch_size=self.config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['per_device_eval_batch_size'],
            weight_decay=self.config['weight_decay'],
            logging_steps=self.config['logging_steps'],
            eval_strategy=self.config['evaluation_strategy'],
            eval_steps=self.config['eval_steps'],
            save_steps=self.config['save_steps'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            dataloader_pin_memory=False
        )

  
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=test_dataset,
            eval_dataset=val_dataset
        )
        
        trainer.train()  

        print('save path is',self.model_training_config.save_trained_model)
        print('save path is',self.model_training_config.save_tokenizer_full_token)

        # Save trained model and tokenizer
        model.save_pretrained(self.model_training_config.save_trained_model)
        tokenizer.save_pretrained(self.model_training_config.save_tokenizer_full_token)

