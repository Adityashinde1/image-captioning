import sys
import io
import numpy as np
from src.entity.config_entity import ModelPredictorConfig
from src.entity.artifacts_entity import ModelTrainerArtifacts, DataPreprocessingArtifacts
from keras.utils.image_utils import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras_preprocessing.sequence import pad_sequences
from src.exception import CustomException
from src.constant import *
from PIL import Image


class ModelPredictor:
    def __init__(self):

        self.model_predictor_config = ModelPredictorConfig()
        self.model_trainer_artifacts = ModelTrainerArtifacts
        self.data_preprocessing_artifacts = DataPreprocessingArtifacts


    # Function to encode a given image into a vector of size (2048, )
    def encode(self, image) -> np.array:
        try:
            #image = self.preprocess_image(image) # preprocess the image
            model = self.model_predictor_config.INCEPTION.inception_model()
            fea_vec = model.predict(image) # Get the encoding vector for the image
            fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
            return fea_vec
        
        except Exception as e:
            raise CustomException(e, sys) from e


    @staticmethod
    def image_caption(photo: np.array, max_length: int, wordtoix: dict, model: object, ixtoword: dict) -> str:
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
            return final

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, image_bytes: bytes):
        try:
            # convert bytes to numpy array
            orig = Image.new(mode='RGB', size=(299,299))
            stream = io.BytesIO(image_bytes)
            orig.save(stream, 'PNG')
            image = Image.open(stream)

            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)

            model = self.model_predictor_config.S3_OPERATION.load_h5_model(BUCKET_NAME,MODEL_NAME,MODEL_NAME)

            encode_img = self.encode(image=image)
            image = encode_img.reshape((1,2048))

            wordtoix = self.model_predictor_config.UTILS.load_pickle_file(filepath=self.model_predictor_config.WORD_TO_INDEX_FILE_PATH)
            ixtoword = self.model_predictor_config.UTILS.load_pickle_file(filepath=self.model_predictor_config.INDEX_TO_WORD_FILE_PATH)

            descriptions = self.model_predictor_config.UTILS.load_pickle_file(filepath=self.model_predictor_config.PREPARED_TRAIN_DESC_PATH)
            max_length = self.model_predictor_config.UTILS.max_length_desc(descriptions=descriptions)
            caption = self.image_caption(photo=image, max_length=max_length, wordtoix=wordtoix, model=model, ixtoword=ixtoword)

            return caption
            
        except Exception as e:
            raise CustomException(e, sys) from e
