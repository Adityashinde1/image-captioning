import os
from datetime import datetime
from from_root import from_root


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_DIR = os.path.join(from_root(), "artifacts", TIMESTAMP)
LOGS_DIR = 'logs'
LOGS_FILE_NAME = 'image_caption.log' 
MODELS_DIR = 'models'

BUCKET_NAME = 'image-captioning-io-files'
S3_DATA_FOLDER_NAME = "data.zip"
S3_TRAIN_TOKEN_FILE_NAME = 'Flickr8k_train_token.txt'
S3_TEST_TOKEN_FILE_NAME = 'Flickr8k_test_token.txt'
S3_TRAIN_IMAGE_NAMES = 'Flickr_8k.trainImages.txt'
S3_TEST_IMAGE_NAMES = 'Flickr_8k.testImages.txt'
S3_GLOVE_MODEL_NAME = 'glove.6B.200d.txt'
GLOVE_MODEL_NAME = 'glove_model.txt'

DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
UNZIP_FOLDER_NAME = 'data/'
TRAIN_TOKEN_FILE_NAME = 'train_token.txt'
TEST_TOKEN_FILE_NAME = 'test_token.txt'

DATA_PREPROCESSING_ARTIFACTS_DIR = "DataPreprocessingArtifacts"
CLEANED_TRAIN_DESC_NAME = 'cleaned_train_desc.txt'
CLEANED_TEST_DESC_NAME = 'cleaned_test_desc.txt'
PREPARED_TRAIN_DESC_FILE_NAME = 'prepared_train_desc.pkl'
EMBEDDING_MATRIX_FILE_NAME = 'embedding_matrix.pkl'
WORD_TO_INDEX_NAME = 'word_to_index.pkl'
INDEX_TO_WORD_NAME = 'index_to_word.pkl'
TRAIN_IMAGE_WITH_PATH_NAME = 'train_img_with_path.pkl'
TEST_IMAGE_WITH_PATH_NAME = 'test_img_with_path.pkl'

WORD_COUNT_THRESHOLD = 4
EMBEDDING_DIM = 200 
LOSS = "categorical_crossentropy"
LEARNING_RATE = 0.01
EPOCHS = 125
NUMBER_OF_PICS_PER_BATCH = 5

MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifats'
INCEPTION_WEIGHT = 'imagenet'
TRAIN_FEATURE_FILE_NAME = 'train_images_features.pkl'
TEST_FEATURE_FILE_NAME = 'test_images_features.pkl'
MODEL_WEIGHT_DIR_NAME = 'model_weights'
MODEL_NAME = 'model.h5'

APP_HOST = "0.0.0.0"
APP_PORT = 8000