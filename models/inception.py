import sys
from src.exception import CustomException
import logging
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from typing import object

logger = logging.getLogger(__name__)

class Inception:
    def __init__(self, weights: str):
        self.inception_v3 = InceptionV3(weights=weights)


    # Creating a new model, by removing the last layer(output layer) from the inception v3
    def inception_model(self) -> object:
        logger.info("Entered the the inception_model method of Inception class")
        try:
            model_new = Model(self.inception_v3.input, self.inception_v3.layers[-2].output)
            logger.info("Created the new model by removing the last output layer frm the inception v3")
            logger.info("Exited the the inception_model method of Inception class")
            return model_new

        except Exception as e:
            raise CustomException(e, sys) from e