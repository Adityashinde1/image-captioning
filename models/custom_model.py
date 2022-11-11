import sys
from src.entity.artifacts_entity import DataPreprocessingArtifacts
from src.exception import CustomException
import numpy as np
from keras import Input
from keras.layers import Dropout, Embedding, LSTM, Dense
from keras.layers.merging import add
from keras.models import Model


class CustomModel:
    def __init__(self, data_preprocessing_artifacts: DataPreprocessingArtifacts):
        self.data_preprocessing_artifacts = data_preprocessing_artifacts

    def main_model(self, max_length: int, vocab_size: int, embedding_dim: int, embedding_matrix: np.array) -> object:
        try:
            # image feature extractor model
            inputs1 = Input(shape=(2048,))
            fe1 = Dropout(0.5)(inputs1)
            fe2 = Dense(256, activation='relu')(fe1)

            # partial caption sequence model
            inputs2 = Input(shape=(max_length,))
            se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
            se2 = Dropout(0.5)(se1)
            se3 = LSTM(256)(se2)

            # decoder (feed forward) model
            decoder1 = add([fe2, se3])
            decoder2 = Dense(256, activation='relu')(decoder1)
            outputs = Dense(vocab_size, activation='softmax')(decoder2)

            # merge the two input models
            model = Model(inputs=[inputs1, inputs2], outputs=outputs)
            
            #We set weights for layers here
            model.layers[2].set_weights([embedding_matrix])
            model.layers[2].trainable = False
            return model

        except Exception as e:
            raise CustomException(e, sys) from e