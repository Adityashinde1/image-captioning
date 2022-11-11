import sys
from src.exception import CustomException
import logging
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model


logger = logging.getLogger(__name__)

class Inception:
    def __init__(self, weights: str):
        self.inception_v3 = InceptionV3(weights=weights)


    # Creating a new model, by removing the last layer(output layer) from the inception v3
    def inception_model(self) -> object:
        try:
            model_new = Model(self.inception_v3.input, self.inception_v3.layers[-2].output)
            return model_new

        except Exception as e:
            raise CustomException(e, sys) from e