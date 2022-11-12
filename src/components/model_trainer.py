import os
import sys
import tensorflow as tf
from src.entity.artifacts_entity import DataPreprocessingArtifacts, DataIngestionArtifacts, ModelTrainerArtifacts
from src.entity.config_entity import ModelTrainerConfig
from src.exception import CustomException
from models.custom_model import CustomModel
from src.constant import *
import logging
from numpy import array
from time import time
import numpy as np
from keras.utils.image_utils import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

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


    # data generator, intended to be used in a call to model.fit_generator()
    @staticmethod
    def data_generator(prepared_descriptions: dict, image_features: dict, wordtoindex: dict, max_length: int, num_of_pics_per_batch: int, vocab_size: int):
        try:
            X1, X2, y = list(), list(), list()
            n=0
            # loop for ever over images
            while 1:
                for key, desc_list in prepared_descriptions.items():
                    n+=1
                    # retrieve the photo feature
                    photo = image_features[key+'.jpg']
                    for desc in desc_list:
                        # encode the sequence
                        seq = [wordtoindex[word] for word in desc.split(' ') if word in wordtoindex]
                        # split one sequence into multiple X, y pairs
                        for i in range(1, len(seq)):
                            # split into input and output pair
                            in_seq, out_seq = seq[:i], seq[i]
                            # pad input sequence
                            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                            # encode output sequence
                            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                            # store
                            X1.append(photo)
                            X2.append(in_seq)
                            y.append(out_seq)
                    # yield the batch data
                    if n==num_of_pics_per_batch:
                        yield [[array(X1), array(X2)], array(y)]
                        X1, X2, y = list(), list(), list()
                        n=0

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

            custom_model = CustomModel(data_preprocessing_artifacts=self.data_preprocessing_artifacts)

            embedding_matrix = self.model_trainer_config.UTILS.load_pickle_file(filepath=self.data_preprocessing_artifacts.embedding_matrix_path)
            model = custom_model.main_model(max_length=self.data_preprocessing_artifacts.max_length, vocab_size=self.data_preprocessing_artifacts.vocab_size,
                                    embedding_dim=EMBEDDING_DIM, embedding_matrix=embedding_matrix)

            model.compile(loss=LOSS, optimizer=tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE))

            train_description = self.model_trainer_config.UTILS.load_pickle_file(filepath=self.data_preprocessing_artifacts.prepared_train_description_path)
            word_to_index = self.model_trainer_config.UTILS.load_pickle_file(filepath=self.data_preprocessing_artifacts.word_to_index_path)

            steps = len(train_description) // NUMBER_OF_PICS_PER_BATCH

            print(">>>>>>>>>>>>>>>>>>>>>>>>> Model training started <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            for epoch in range(EPOCHS):
                generator = self.data_generator(prepared_descriptions=train_description,image_features=train_image_array, wordtoindex=word_to_index, 
                                                max_length=self.data_preprocessing_artifacts.max_length,num_of_pics_per_batch=NUMBER_OF_PICS_PER_BATCH, 
                                                vocab_size=self.data_preprocessing_artifacts.vocab_size)
                model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
            model.save(self.model_trainer_config.MODEL_WEIGHT_PATH)

            model_trainer_artifacts = ModelTrainerArtifacts(train_image_features_path=self.model_trainer_config.TRAIN_FEATURE_PATH,
                                                            test_image_features_path=self.model_trainer_config.TEST_FEATURE_PATH,
                                                            trained_model_path=self.model_trainer_config.MODEL_WEIGHT_PATH)

            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e