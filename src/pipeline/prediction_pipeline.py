import sys
import io
import logging
import numpy as np
from src.entity.config_entity import ModelPredictorConfig
from src.entity.artifacts_entity import ModelTrainerArtifacts, DataPreprocessingArtifacts
from keras.utils.image_utils import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras_preprocessing.sequence import pad_sequences
from src.exception import CustomException
from src.constant import *
from PIL import Image

logger = logging.getLogger(__name__)


class ModelPredictor:
    def __init__(self):

        self.model_predictor_config = ModelPredictorConfig()
        self.model_trainer_artifacts = ModelTrainerArtifacts
        self.data_preprocessing_artifacts = DataPreprocessingArtifacts


    # Function to encode a given image into a vector of size (2048, )
    def encode(self, image) -> np.array:
        logger.info("Entered the encode method of Model predictor class.")
        try:
            #image = self.preprocess_image(image) # preprocess the image
            model = self.model_predictor_config.INCEPTION.inception_model()
            fea_vec = model.predict(image) # Get the encoding vector for the image
            fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
            logger.info("Exited the encode method of Model predictor class.")
            return fea_vec
        
        except Exception as e:
            raise CustomException(e, sys) from e


    @staticmethod
    def image_caption(photo: np.array, max_length: int, wordtoix: dict, model: object, ixtoword: dict) -> str:
        logger.info("Entered the image_caption method of Model predictor class.")
        try:
            in_text = 'startseq'
            for i in range(max_length):
                sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
                sequence = pad_sequences([sequence], maxlen=max_length)
                yhat = model.predict([photo,sequence], verbose=0)
                yhat = np.argmax(yhat)
                word = ixtoword[yhat]
                in_text += ' ' + word
                if word == 'endseq':
                    break
            final = in_text.split()
            final = final[1:-1]
            final = ' '.join(final)
            logger.info("Exited the image_caption method of Model predictor class.")
            return final

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, image_bytes: bytes):
        logger.info("Entered the run_pipeline method of Model predictor class.")
        try:
            # Convert bytes to numpy array
            orig = Image.new(mode='RGB', size=(299,299))
            stream = io.BytesIO(image_bytes)
            orig.save(stream, 'PNG')
            image = Image.open(stream)

            # Converting image to array
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)

            # Loading model
            model = self.model_predictor_config.S3_OPERATION.load_h5_model(BUCKET_NAME,MODEL_NAME,MODEL_NAME)
            logger.info("Model loaded from s3 bucket for prediction")

            encode_img = self.encode(image=image)
            image = encode_img.reshape((1,2048))

            # Loading corpus with index
            wordtoix = self.model_predictor_config.UTILS.load_pickle_file(filepath=self.model_predictor_config.WORD_TO_INDEX_FILE_PATH)
            ixtoword = self.model_predictor_config.UTILS.load_pickle_file(filepath=self.model_predictor_config.INDEX_TO_WORD_FILE_PATH)

            descriptions = self.model_predictor_config.UTILS.load_pickle_file(filepath=self.model_predictor_config.PREPARED_TRAIN_DESC_PATH)
            max_length = self.model_predictor_config.UTILS.max_length_desc(descriptions=descriptions)
            
            # Generating caption for uploaded image by user
            caption = self.image_caption(photo=image, max_length=max_length, wordtoix=wordtoix, model=model, ixtoword=ixtoword)
            logger.info("Exited the run_pipeline method of Model predictor class.")
            return caption
            
        except Exception as e:
            raise CustomException(e, sys) from e
