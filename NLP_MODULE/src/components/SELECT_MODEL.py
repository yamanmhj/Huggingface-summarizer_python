import keras
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM,Activation,Dense,Dropout,Input,Embedding,SpatialDropout1D
from NLP_MODULE.config import LoadConfig
from logger import logging

class SelectModelArchitecture:
    def __init__(self):
        pass

    def get_model_architecture(self):
        
        try:
            config = LoadConfig.get_config_Full_file()
            
            model = Sequential()
            model.add(Embedding(input_dim=int(config['MAX_WORDS']), 
                                output_dim=int(config['EMBEDDING_DIM'])))
            model.add(SpatialDropout1D(float(config['DROP_OUT'])))
            model.add(LSTM(units=int(config['LSTM_OUTPUT_SIZE']), 
                           dropout=float(config['DROP_OUT']),
                           recurrent_dropout=float(config['RECURRENT_DROPOUT'])))
            model.add(Dense(1, activation=config['ACTIVATION']))
            model.compile(loss=config['LOSS'], 
                          optimizer=RMSprop(), 
                          metrics=[config['METRICES']])
            print("Model name returned --------------------------------")
            return model
        except Exception as e:
            
            raise Exception("Error in getting model architecture: ", e)

