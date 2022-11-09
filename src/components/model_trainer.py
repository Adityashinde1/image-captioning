import os
import sys
from src.entity.artifacts_entity import DataTransformationArtifacts
from src.entity.config_entity import ModelTrainerConfig
from src.exception import CustomException
import logging
import numpy as np
from keras.utils.image_utils import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config


    # We're converting our image size 299x299
    @staticmethod
    def preprocess_image(image_path) -> np.array:
        logger.info("Entered the preprocess_image method")
        try:
            # Convert all the images to size 299x299 as expected by the inception v3 model
            img = load_img(image_path, target_size=(299, 299))
            # Convert PIL image to numpy array of 3-dimensions
            x = img_to_array(img)
            # Add one more dimension
            x = np.expand_dims(x, axis=0)
            # preprocess the images using preprocess_input() from inception module
            x = preprocess_input(x)
            logger.info("Exited the preprocess_image method")
            return x
        except Exception as e:
            raise CustomException(e, sys) from e