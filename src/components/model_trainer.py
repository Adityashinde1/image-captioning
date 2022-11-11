import os
import sys
from src.entity.artifacts_entity import DataPreprocessingArtifacts, DataIngestionArtifacts, ModelTrainerArtifacts
from src.entity.config_entity import ModelTrainerConfig
from src.exception import CustomException
import logging
from time import time
import numpy as np
from keras.utils.image_utils import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, data_preprocessing_artifacts: DataPreprocessingArtifacts, 
                model_trainer_config: ModelTrainerConfig, 
                data_ingestion_artifacts: DataIngestionArtifacts):
        self.data_preprocessing_artifacts = data_preprocessing_artifacts
        self.model_trainer_config = model_trainer_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    # We're converting our image size 299x299
    @staticmethod
    def preprocess_image(image_path: str) -> np.array:

        try:
            # Convert all the images to size 299x299 as expected by the inception v3 model
            img = load_img(image_path, target_size=(299, 299))
            # Convert PIL image to numpy array of 3-dimensions
            x = img_to_array(img)
            # Add one more dimension
            x = np.expand_dims(x, axis=0)
            # preprocess the images using preprocess_input() from inception module
            x = preprocess_input(x)
 
            return x

        except Exception as e:
            raise CustomException(e, sys) from e

    # Function to encode a given image into a vector of size (2048, )
    def encode(self, image: str) -> np.array:
        try:
            image = self.preprocess_image(image) # preprocess the image
            model = self.model_trainer_config.INCEPTION.inception_model()
            fea_vec = model.predict(image) # Get the encoding vector for the image
            fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
            return fea_vec

        except Exception as e:
            raise CustomException(e, sys) from e

    def generate_train_img_feature(self, image_data: list, train_image: list) -> dict:
        try:
            start = time()
            train_feature = {}
            for img in train_image:
                train_feature[img[len(image_data):]] = self.encode(img)
            logger.info(f"Time taken for getting the train features - {time()-start}")
            return train_feature

        except Exception as e:
            raise CustomException(e, sys) from e


    def generate_test_img_feature(self, image_data: list, test_image: list) -> dict:
        try:
            start = time()
            test_feature = {}
            for img in test_image:
                test_feature[img[len(image_data):]] = self.encode(img)
            logger.info(f"Time taken for getting the test features - {time()-start}")
            return test_feature

        except Exception as e:
            raise CustomException(e, sys) from e




    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        try:
            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)

            train_image = self.model_trainer_config.UTILS.load_pickle_file(filepath=self.data_preprocessing_artifacts.train_image_path)
            train_image_array = self.generate_train_img_feature(image_data=self.data_ingestion_artifacts.image_data_dir, train_image=train_image)

            test_image = self.model_trainer_config.UTILS.load_pickle_file(filepath=self.data_preprocessing_artifacts.test_image_path)
            test_image_array = self.generate_test_img_feature(image_data=self.data_ingestion_artifacts.image_data_dir, test_image=test_image)

            self.model_trainer_config.UTILS.dump_pickle_file(output_filepath=self.model_trainer_config.TRAIN_FEATURE_PATH, data=train_image_array)
            self.model_trainer_config.UTILS.dump_pickle_file(output_filepath=self.model_trainer_config.TEST_FEATURE_PATH, data=test_image_array)

        except Exception as e:
            raise CustomException(e, sys) from e